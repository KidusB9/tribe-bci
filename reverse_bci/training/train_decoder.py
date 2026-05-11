"""
Training script for the Reverse Decoder + Text Decoder.

Trains the reverse decoder (latent -> text features) and
text decoder (text features -> words) components.

Phase 1: Train reverse decoder on synthetic latent-to-text pairs
Phase 2: Train text decoder on (text_features, word_label) pairs
Phase 3: End-to-end fine-tuning of the full pipeline
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
import logging

from reverse_bci.reverse_decoder import ReverseDecoder, ReverseDecoderConfig
from reverse_bci.text_decoder import TextDecoder, TextDecoderConfig
from reverse_bci.training.dataset import SyntheticPairedDataset

logger = logging.getLogger(__name__)


class DecoderTrainer:
    """Trains the reverse decoder and text decoder components.

    Usage:
        trainer = DecoderTrainer()
        trainer.train_reverse_decoder(tribe_model)
        trainer.train_text_decoder()
        trainer.train_end_to_end(eeg_encoder, adapter)
        trainer.save("decoder_checkpoint.pt")
    """

    def __init__(
        self,
        decoder_config: Optional[ReverseDecoderConfig] = None,
        text_config: Optional[TextDecoderConfig] = None,
        lr: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 50,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.decoder = ReverseDecoder(decoder_config or ReverseDecoderConfig()).to(self.device)
        self.text_decoder = TextDecoder(text_config or TextDecoderConfig()).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs

    def train_reverse_decoder(
        self,
        tribe_model=None,
        n_augmentations: int = 20,
        cache_dir: str = "./cache/training",
    ):
        """Phase 1: Train reverse decoder to recover text features from latents."""
        logger.info("Phase 1: Training reverse decoder")

        # Initialize from TRIBE weights if available
        if tribe_model is not None:
            model = tribe_model._model if hasattr(tribe_model, "_model") else tribe_model
            self.decoder.init_from_tribe(model)

        dataset = SyntheticPairedDataset(
            tribe_model=tribe_model,
            n_augmentations=n_augmentations,
            cache_dir=cache_dir,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        optimizer = optim.AdamW(
            self.decoder.parameters(), lr=self.lr, weight_decay=0.01,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        for epoch in range(self.n_epochs):
            self.decoder.train()
            epoch_loss = 0.0

            for batch in loader:
                latent = batch["latent"].to(self.device)
                target_features = batch["text_features"].to(self.device)

                # Decode
                output = self.decoder(latent)
                pred_features = output["text_features"]

                # Feature reconstruction loss
                feature_loss = nn.functional.mse_loss(pred_features, target_features)

                # Cosine similarity loss (features should point same direction)
                cos_loss = 1.0 - nn.functional.cosine_similarity(
                    pred_features, target_features, dim=-1
                ).mean()

                total_loss = feature_loss + 0.5 * cos_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Reverse decoder - Epoch %d/%d - Loss: %.4f",
                    epoch + 1, self.n_epochs, avg_loss,
                )

    def train_text_decoder(
        self,
        tribe_model=None,
        n_augmentations: int = 20,
        cache_dir: str = "./cache/training",
    ):
        """Phase 2: Train text decoder to map text features to words."""
        logger.info("Phase 2: Training text decoder")

        dataset = SyntheticPairedDataset(
            tribe_model=tribe_model,
            n_augmentations=n_augmentations,
            cache_dir=cache_dir,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        # Only train the text decoder's neural and vocabulary components
        params = list(self.text_decoder.parameters())
        optimizer = optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        for epoch in range(self.n_epochs):
            self.text_decoder.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                text_features = batch["text_features"].to(self.device)
                labels = batch["label"].to(self.device)

                # Decode through the text decoder
                output = self.text_decoder(text_features)

                # Classification loss on neural decoder logits
                if output["neural_logits"] is not None:
                    # Ensure labels are in range
                    valid_mask = labels < output["neural_logits"].shape[-1]
                    if valid_mask.any():
                        cls_loss = nn.functional.cross_entropy(
                            output["neural_logits"][valid_mask],
                            labels[valid_mask],
                        )
                        preds = output["neural_logits"][valid_mask].argmax(dim=-1)
                        correct += (preds == labels[valid_mask]).sum().item()
                        total += valid_mask.sum().item()
                    else:
                        cls_loss = torch.tensor(0.0, device=self.device)
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)

                # Vocabulary embedding loss
                if hasattr(self.text_decoder, "vocab_embed"):
                    vocab_embeds = self.text_decoder.vocab_embed()
                    valid_labels = labels[labels < len(self.text_decoder.vocab)]
                    if len(valid_labels) > 0:
                        target_embeds = vocab_embeds[valid_labels]
                        valid_features = text_features[labels < len(self.text_decoder.vocab)]
                        embed_loss = 1.0 - nn.functional.cosine_similarity(
                            self.text_decoder.vocab_embed.proj(valid_features),
                            self.text_decoder.vocab_embed.proj(target_embeds),
                            dim=-1,
                        ).mean()
                    else:
                        embed_loss = torch.tensor(0.0, device=self.device)
                else:
                    embed_loss = torch.tensor(0.0, device=self.device)

                total_loss = cls_loss + 0.5 * embed_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                epoch_loss += total_loss.item()

            scheduler.step()
            avg_loss = epoch_loss / max(len(loader), 1)
            accuracy = correct / max(total, 1) * 100

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Text decoder - Epoch %d/%d - Loss: %.4f - Acc: %.1f%%",
                    epoch + 1, self.n_epochs, avg_loss, accuracy,
                )

    def train_end_to_end(
        self,
        encoder: nn.Module,
        adapter: nn.Module,
        eeg_dataset,
        tribe_model=None,
        n_epochs: Optional[int] = None,
    ):
        """Phase 3: End-to-end fine-tuning of the entire pipeline.

        EEG -> Encoder -> Adapter -> Reverse Decoder -> Text Decoder -> Word
        All components trained jointly with a classification objective.
        """
        n_epochs = n_epochs or self.n_epochs // 2
        logger.info("Phase 3: End-to-end fine-tuning for %d epochs", n_epochs)

        loader = DataLoader(
            eeg_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        all_params = (
            list(encoder.parameters()) +
            list(adapter.parameters()) +
            list(self.decoder.parameters()) +
            list(self.text_decoder.parameters())
        )
        optimizer = optim.AdamW(all_params, lr=self.lr * 0.1, weight_decay=0.01)

        for epoch in range(n_epochs):
            encoder.train()
            adapter.train()
            self.decoder.train()
            self.text_decoder.train()

            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                eeg = batch["eeg"].to(self.device)
                labels = batch["label"].to(self.device)

                # Full pipeline forward pass
                enc_out = encoder(eeg)
                adapted = adapter(enc_out["latent"])
                dec_out = self.decoder(adapted)
                text_out = self.text_decoder(dec_out["text_features"])

                # Classification loss
                if text_out["neural_logits"] is not None:
                    valid = labels < text_out["neural_logits"].shape[-1]
                    if valid.any():
                        loss = nn.functional.cross_entropy(
                            text_out["neural_logits"][valid], labels[valid],
                        )
                        preds = text_out["neural_logits"][valid].argmax(-1)
                        correct += (preds == labels[valid]).sum().item()
                        total += valid.sum().item()
                    else:
                        loss = torch.tensor(0.0, device=self.device)
                else:
                    loss = torch.tensor(0.0, device=self.device)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            accuracy = correct / max(total, 1) * 100

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "End-to-end - Epoch %d/%d - Loss: %.4f - Acc: %.1f%%",
                    epoch + 1, n_epochs, avg_loss, accuracy,
                )

    def save(self, path: str):
        state = {
            "decoder": self.decoder.state_dict(),
            "text_decoder": self.text_decoder.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        logger.info("Saved decoder checkpoint to %s", path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.decoder.load_state_dict(state["decoder"])
        self.text_decoder.load_state_dict(state["text_decoder"])
        logger.info("Loaded decoder checkpoint from %s", path)
