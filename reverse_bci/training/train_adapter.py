"""
Training script for the Domain Adapter.

Trains the EEG encoder + domain adapter to map EEG signals into
TRIBE v2's latent space. Supports multiple training phases:

Phase 1: Self-supervised pre-training on EEG data alone
Phase 2: Synthetic paired training using TRIBE v2 as teacher
Phase 3: Fine-tuning with real EEG + stimulus paired data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path
import logging
import time

from reverse_bci.eeg_encoder import EEGEncoder, EEGEncoderConfig
from reverse_bci.domain_adapter import DomainAdapter, DomainAdapterConfig
from reverse_bci.training.dataset import BCIDataset, SyntheticPairedDataset, EEGAugmentation

logger = logging.getLogger(__name__)


class AdapterTrainer:
    """Trains the EEG encoder and domain adapter.

    Usage:
        trainer = AdapterTrainer(config)
        trainer.train_synthetic(tribe_model)  # Phase 2
        trainer.train_eeg(eeg_dataset)        # Phase 3
        trainer.save("adapter_checkpoint.pt")
    """

    def __init__(
        self,
        encoder_config: Optional[EEGEncoderConfig] = None,
        adapter_config: Optional[DomainAdapterConfig] = None,
        lr: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 50,
        device: str = "auto",
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.encoder = EEGEncoder(encoder_config or EEGEncoderConfig()).to(self.device)
        self.adapter = DomainAdapter(adapter_config or DomainAdapterConfig()).to(self.device)
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        self.optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.adapter.parameters()),
            lr=lr, weight_decay=0.01,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=n_epochs,
        )

    def train_synthetic(
        self,
        tribe_model=None,
        n_augmentations: int = 20,
        cache_dir: str = "./cache/training",
    ):
        """Phase 2: Train with synthetic paired data from TRIBE v2.

        Uses TRIBE v2 to generate (latent, text_features) pairs,
        then trains the reverse path to recover text features from latents.
        """
        logger.info("Phase 2: Synthetic paired training")

        dataset = SyntheticPairedDataset(
            tribe_model=tribe_model,
            n_augmentations=n_augmentations,
            cache_dir=cache_dir,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.adapter.train()

            epoch_loss = 0.0
            n_batches = 0

            for batch in loader:
                latent = batch["latent"].to(self.device)
                text_features = batch["text_features"].to(self.device)
                labels = batch["label"].to(self.device)

                # Simulate EEG encoding by adding noise to the latent
                # (In real training, this would come from the EEG encoder)
                noise = torch.randn_like(latent) * 0.3
                noisy_latent = latent + noise

                # Domain adaptation
                adapted = self.adapter(noisy_latent)

                # Loss: adapted latent should be close to original
                reconstruction_loss = nn.functional.mse_loss(adapted, latent)

                # Contrastive loss if we have TRIBE latents
                losses = self.adapter.compute_total_loss(
                    eeg_latent=noisy_latent,
                    tribe_latent=latent,
                )

                total_loss = losses["total"] + reconstruction_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.adapter.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            self.scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    "Epoch %d/%d - Loss: %.4f - LR: %.2e",
                    epoch + 1, self.n_epochs, avg_loss,
                    self.scheduler.get_last_lr()[0],
                )

    def train_eeg(
        self,
        dataset: BCIDataset,
        tribe_model=None,
        val_dataset: Optional[BCIDataset] = None,
    ):
        """Phase 3: Fine-tune with real EEG data.

        Requires EEG recordings paired with known stimuli.
        Optionally uses TRIBE v2 to generate target latents.
        """
        logger.info("Phase 3: EEG fine-tuning with %d trials", len(dataset))

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        best_val_loss = float("inf")

        for epoch in range(self.n_epochs):
            self.encoder.train()
            self.adapter.train()

            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch in loader:
                eeg = batch["eeg"].to(self.device)  # (B, C, T)
                labels = batch["label"].to(self.device)

                # Encode EEG
                enc_output = self.encoder(eeg)
                eeg_latent = enc_output["latent"]

                # Adapt to TRIBE space
                adapted = self.adapter(eeg_latent)

                # Classification loss (proxy for decoding quality)
                classifier_logits = torch.matmul(
                    adapted,
                    self.adapter.refiner.refine[0].weight[:adapted.shape[-1]].T,
                )
                if classifier_logits.shape[-1] >= labels.max() + 1:
                    cls_loss = nn.functional.cross_entropy(
                        classifier_logits[:, :labels.max() + 1], labels
                    )
                else:
                    cls_loss = torch.tensor(0.0, device=self.device)

                # Reconstruction regularization
                reg_loss = 0.01 * adapted.pow(2).mean()

                total_loss = cls_loss + reg_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.adapter.parameters()),
                    max_norm=1.0,
                )
                self.optimizer.step()

                epoch_loss += total_loss.item()

            self.scheduler.step()

            avg_loss = epoch_loss / max(len(loader), 1)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info("Epoch %d/%d - Loss: %.4f", epoch + 1, self.n_epochs, avg_loss)

            # Validation
            if val_dataset is not None and (epoch + 1) % 5 == 0:
                val_loss = self._validate(val_dataset)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    logger.info("New best validation loss: %.4f", val_loss)

    @torch.no_grad()
    def _validate(self, dataset: BCIDataset) -> float:
        self.encoder.eval()
        self.adapter.eval()
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        total_loss = 0.0
        for batch in loader:
            eeg = batch["eeg"].to(self.device)
            labels = batch["label"].to(self.device)
            enc_output = self.encoder(eeg)
            adapted = self.adapter(enc_output["latent"])
            loss = adapted.pow(2).mean()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)

    def save(self, path: str):
        state = {
            "encoder": self.encoder.state_dict(),
            "adapter": self.adapter.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, path)
        logger.info("Saved adapter checkpoint to %s", path)

    def load(self, path: str):
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.encoder.load_state_dict(state["encoder"])
        self.adapter.load_state_dict(state["adapter"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        logger.info("Loaded adapter checkpoint from %s", path)
