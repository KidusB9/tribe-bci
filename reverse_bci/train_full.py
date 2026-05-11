#!/usr/bin/env python3
"""
Full training pipeline for the Reverse BCI system.

This script trains all components of the reverse BCI:
1. Domain adapter (EEG -> TRIBE latent space)
2. Reverse decoder (latent -> text features)
3. Text decoder (text features -> words)
4. End-to-end fine-tuning

Usage:
    # Train with synthetic data only (no hardware needed)
    python -m reverse_bci.train_full

    # Train with TRIBE v2 initialization (much better)
    python -m reverse_bci.train_full --tribe facebook/tribev2

    # Train with real EEG data
    python -m reverse_bci.train_full --tribe facebook/tribev2 --eeg-data ./eeg_recordings/

    # Resume from checkpoint
    python -m reverse_bci.train_full --resume ./checkpoints/latest.pt
"""

import argparse
import logging
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reverse_bci.train")


def main():
    parser = argparse.ArgumentParser(description="Train Reverse BCI")
    parser.add_argument(
        "--tribe", default=None,
        help="TRIBE v2 model path or HuggingFace repo (e.g. facebook/tribev2)",
    )
    parser.add_argument(
        "--eeg-data", default=None,
        help="Path to real EEG recording data (.npz or directory)",
    )
    parser.add_argument(
        "--output", default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--headset", default="muse",
        choices=["muse", "emotiv_epoc", "emotiv_insight", "openbci_cyton"],
        help="Target headset (determines channel count)",
    )

    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info("Using device: %s", device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TRIBE v2 if specified
    tribe_model = None
    if args.tribe:
        logger.info("Loading TRIBE v2 from %s...", args.tribe)
        from tribev2 import TribeModel
        tribe_model = TribeModel.from_pretrained(args.tribe, device=device)
        logger.info("TRIBE v2 loaded")

    # Configure for target headset
    from reverse_bci.eeg_processor import HEADSET_MONTAGES
    from reverse_bci.eeg_encoder import EEGEncoderConfig
    from reverse_bci.domain_adapter import DomainAdapterConfig
    from reverse_bci.reverse_decoder import ReverseDecoderConfig
    from reverse_bci.text_decoder import TextDecoderConfig

    montage = HEADSET_MONTAGES[args.headset]
    encoder_config = EEGEncoderConfig(
        n_channels=montage["n_channels"],
        n_samples=int(2.0 * montage["sfreq"]),  # 2-second windows
        latent_dim=1152,
    )

    # ----------------------------------------------------------------
    # Phase 1: Train the reverse decoder
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 1: Training Reverse Decoder")
    logger.info("=" * 60)

    from reverse_bci.training.train_decoder import DecoderTrainer

    decoder_trainer = DecoderTrainer(
        decoder_config=ReverseDecoderConfig(),
        text_config=TextDecoderConfig(),
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=device,
    )

    decoder_trainer.train_reverse_decoder(
        tribe_model=tribe_model,
        n_augmentations=20,
        cache_dir=str(output_dir / "cache"),
    )

    # ----------------------------------------------------------------
    # Phase 2: Train the text decoder
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Training Text Decoder")
    logger.info("=" * 60)

    decoder_trainer.train_text_decoder(
        tribe_model=tribe_model,
        n_augmentations=20,
        cache_dir=str(output_dir / "cache"),
    )

    decoder_trainer.save(str(output_dir / "decoder_checkpoint.pt"))

    # ----------------------------------------------------------------
    # Phase 3: Train the domain adapter
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 3: Training Domain Adapter")
    logger.info("=" * 60)

    from reverse_bci.training.train_adapter import AdapterTrainer

    adapter_trainer = AdapterTrainer(
        encoder_config=encoder_config,
        adapter_config=DomainAdapterConfig(),
        lr=args.lr,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        device=device,
    )

    adapter_trainer.train_synthetic(
        tribe_model=tribe_model,
        n_augmentations=20,
        cache_dir=str(output_dir / "cache"),
    )

    # ----------------------------------------------------------------
    # Phase 4: Fine-tune with real EEG data (if available)
    # ----------------------------------------------------------------
    if args.eeg_data:
        logger.info("=" * 60)
        logger.info("Phase 4: Fine-tuning with real EEG data")
        logger.info("=" * 60)

        from reverse_bci.training.dataset import BCIDataset, EEGAugmentation

        eeg_dataset = BCIDataset(
            data_path=args.eeg_data,
            transform=EEGAugmentation(),
        )

        adapter_trainer.train_eeg(eeg_dataset, tribe_model=tribe_model)

        # End-to-end fine-tuning
        logger.info("Phase 4b: End-to-end fine-tuning")
        decoder_trainer.train_end_to_end(
            encoder=adapter_trainer.encoder,
            adapter=adapter_trainer.adapter,
            eeg_dataset=eeg_dataset,
            tribe_model=tribe_model,
            n_epochs=args.epochs // 2,
        )

    adapter_trainer.save(str(output_dir / "adapter_checkpoint.pt"))

    # ----------------------------------------------------------------
    # Save combined checkpoint
    # ----------------------------------------------------------------
    combined_state = {
        "encoder": adapter_trainer.encoder.state_dict(),
        "adapter": adapter_trainer.adapter.state_dict(),
        "decoder": decoder_trainer.decoder.state_dict(),
        "text_decoder": decoder_trainer.text_decoder.state_dict(),
    }
    combined_path = output_dir / "bci_model.pt"
    torch.save(combined_state, combined_path)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Combined model saved to: %s", combined_path)
    logger.info("=" * 60)
    logger.info("")
    logger.info("To run live decoding:")
    logger.info("  python -m reverse_bci.ui.app --model %s --headset %s", combined_path, args.headset)


if __name__ == "__main__":
    main()
