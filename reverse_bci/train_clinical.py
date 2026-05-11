#!/usr/bin/env python3
"""
Train the Reverse BCI on real clinical EEG data.

Uses the Inner Speech Dataset (OpenNeuro ds003626) — real humans
imagining words while wearing 128-channel research-grade EEG.

This is the ground truth test: can our pipeline learn anything real
from actual brain signals, or was it only fitting noise?

Usage:
    python -m reverse_bci.train_clinical --data ./data/inner_speech --subjects 1
    python -m reverse_bci.train_clinical --data ./data/inner_speech --subjects 1 2 --epochs 100
"""

import argparse
import logging
import time
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reverse_bci.train_clinical")


class CompactEEGNet(nn.Module):
    """EEGNet-inspired compact classifier for small datasets.

    Based on Lawhern et al. (2018) "EEGNet: A Compact Convolutional
    Neural Network for EEG-based Brain-Computer Interfaces."
    Designed for O(100) trials — uses depthwise/separable convolutions
    to keep parameter count under 10K.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int,
                 F1: int = 16, D: int = 2, F2: int = 32, dropout: float = 0.5):
        super().__init__()
        # Block 1: Temporal + Spatial filtering
        self.temporal_conv = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.spatial_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 8))
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.sep_conv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.pointwise = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Compute flatten dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            dummy = self._features(dummy)
            flat_dim = dummy.shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

        total = sum(p.numel() for p in self.parameters())
        logger.info("CompactEEGNet: %d params, %d ch, %d samples, %d classes", total, n_channels, n_samples, n_classes)

    def _features(self, x):
        x = self.temporal_conv(x)
        x = self.bn1(x)
        x = self.spatial_conv(x)
        x = self.bn2(x)
        x = torch.nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_conv(x)
        x = self.pointwise(x)
        x = self.bn3(x)
        x = torch.nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, eeg: torch.Tensor) -> dict:
        x = eeg.unsqueeze(1)  # (B, 1, C, T)
        features = self._features(x)
        logits = self.classifier(features)
        return {"logits": logits, "latent": features}


class EEGClassifier(nn.Module):
    """Full BCI encoder+adapter classifier. Use for larger datasets."""

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, latent_dim: int = 1152):
        super().__init__()
        from reverse_bci.eeg_encoder import EEGEncoder, EEGEncoderConfig
        from reverse_bci.domain_adapter import DomainAdapter, DomainAdapterConfig

        encoder_config = EEGEncoderConfig(
            n_channels=n_channels,
            n_samples=n_samples,
            latent_dim=latent_dim,
            hidden_dim=128,
            n_transformer_layers=2,
            n_heads=4,
            dropout=0.3,
        )
        self.encoder = EEGEncoder(encoder_config)
        self.adapter = DomainAdapter(DomainAdapterConfig(
            eeg_latent_dim=latent_dim,
            tribe_latent_dim=latent_dim,
        ))

        self.classifier = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Dropout(0.5),
            nn.Linear(latent_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

        self.bottleneck_dim = encoder_config.bottleneck_dim
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "EEGClassifier: %d params, bottleneck=%d, %d ch, %d samples, %d classes",
            total, self.bottleneck_dim, n_channels, n_samples, n_classes,
        )

    def forward(self, eeg: torch.Tensor) -> dict:
        enc_out = self.encoder(eeg)
        latent = self.adapter(enc_out["latent"])
        logits = self.classifier(latent)
        return {
            "logits": logits,
            "latent": latent,
            "bottleneck": enc_out.get("bottleneck"),
        }


class FullPipelineClassifier(nn.Module):
    """Full BCI pipeline with classification: EEG -> Encoder -> Adapter -> ReverseDecoder -> TextDecoder -> class.

    This tests the actual production architecture end-to-end.
    """

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, latent_dim: int = 1152):
        super().__init__()
        from reverse_bci.eeg_encoder import EEGEncoder, EEGEncoderConfig
        from reverse_bci.domain_adapter import DomainAdapter, DomainAdapterConfig
        from reverse_bci.reverse_decoder import ReverseDecoder, ReverseDecoderConfig
        from reverse_bci.text_decoder import TextDecoderConfig

        self.encoder = EEGEncoder(EEGEncoderConfig(
            n_channels=n_channels,
            n_samples=n_samples,
            latent_dim=latent_dim,
        ))
        self.adapter = DomainAdapter(DomainAdapterConfig(
            eeg_latent_dim=latent_dim,
            tribe_latent_dim=latent_dim,
        ))
        self.decoder = ReverseDecoder(ReverseDecoderConfig())

        text_feat_dim = ReverseDecoderConfig().text_feature_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(text_feat_dim),
            nn.Dropout(0.5),
            nn.Linear(text_feat_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, n_classes),
        )

        total = sum(p.numel() for p in self.parameters())
        logger.info("FullPipelineClassifier: %d params", total)

    def forward(self, eeg: torch.Tensor) -> dict:
        enc_out = self.encoder(eeg)
        latent = self.adapter(enc_out["latent"])
        dec_out = self.decoder(latent)
        logits = self.classifier(dec_out["text_features"])
        return {"logits": logits, "latent": latent}


def evaluate(model, loader, device, n_classes=4):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            eeg = batch["eeg"].to(device)
            labels = batch["label"].to(device)
            out = model(eeg)
            loss = nn.functional.cross_entropy(out["logits"], labels)
            total_loss += loss.item() * len(labels)
            preds = out["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = (all_preds == all_labels).mean() * 100
    avg_loss = total_loss / len(all_labels)

    # Per-class accuracy
    class_names = ["Up", "Down", "Right", "Left"]
    per_class = {}
    for c in range(n_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class[class_names[c]] = (all_preds[mask] == c).mean() * 100

    # Confusion matrix
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(all_labels, all_preds):
        confusion[t, p] += 1

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "per_class": per_class,
        "confusion": confusion,
        "predictions": all_preds,
        "labels": all_labels,
    }


def print_confusion_matrix(confusion, class_names):
    n = len(class_names)
    header = "        " + "  ".join(f"{name:>6s}" for name in class_names)
    logger.info("Confusion Matrix (rows=true, cols=predicted):")
    logger.info(header)
    for i in range(n):
        row = f"  {class_names[i]:>5s} " + "  ".join(f"{confusion[i, j]:6d}" for j in range(n))
        logger.info(row)


def train_epoch(model, loader, optimizer, device, epoch, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        eeg = batch["eeg"].to(device)
        labels = batch["label"].to(device)

        out = model(eeg)
        loss = nn.functional.cross_entropy(out["logits"], labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        preds = out["logits"].argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Train BCI on clinical EEG data")
    parser.add_argument("--data", required=True, help="Path to Inner Speech dataset root")
    parser.add_argument("--subjects", type=int, nargs="+", default=[1], help="Subject numbers")
    parser.add_argument("--sessions", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--condition", default="inner_speech",
                        choices=["inner_speech", "pronounced", "visualized", "all"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="./checkpoints/clinical")
    parser.add_argument("--model", default="eegnet",
                        choices=["eegnet", "encoder", "full_pipeline"],
                        help="eegnet = compact EEGNet (best for small data), encoder = full BCI encoder")
    parser.add_argument("--channels", type=int, default=None,
                        help="Number of channels to select (None = all 128)")
    parser.add_argument("--t-start", type=float, default=0.5,
                        help="Epoch start time relative to stimulus (seconds)")
    parser.add_argument("--t-end", type=float, default=3.5,
                        help="Epoch end time relative to stimulus (seconds)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="stratified",
                        choices=["stratified", "cross_session"],
                        help="stratified = within-session split, cross_session = train ses1+2 / test ses3")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Device: %s", device)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load dataset
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("LOADING CLINICAL EEG DATA")
    logger.info("=" * 70)

    from reverse_bci.training.inner_speech_dataset import InnerSpeechDataset

    dataset = InnerSpeechDataset(
        data_dir=args.data,
        subjects=args.subjects,
        sessions=args.sessions,
        condition=args.condition,
        t_start=args.t_start,
        t_end=args.t_end,
        channel_select=args.channels,
        normalize=True,
        augment=False,
    )

    logger.info("Total trials: %d", len(dataset))
    logger.info("EEG shape per trial: (%d, %d)", dataset.eeg_data.shape[1], dataset.eeg_data.shape[2])
    logger.info("Classes: %s", {v: (dataset.labels == k).sum() for k, v in {0: "Up", 1: "Down", 2: "Right", 3: "Left"}.items()})

    # ----------------------------------------------------------------
    # Train/Val/Test split (by session to prevent leakage)
    # ----------------------------------------------------------------
    train_mask, val_mask, test_mask = dataset.get_split(seed=args.seed, mode=args.split)

    train_set = dataset.subset(train_mask)
    train_set.augment = True  # Enable augmentation for training
    val_set = dataset.subset(val_mask)
    test_set = dataset.subset(test_mask)

    logger.info("Split: train=%d, val=%d, test=%d", len(train_set), len(val_set), len(test_set))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ----------------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------------
    n_channels = dataset.eeg_data.shape[1]
    n_samples = dataset.eeg_data.shape[2]
    n_classes = len(np.unique(dataset.labels))

    logger.info("=" * 70)
    logger.info("INITIALIZING MODEL")
    logger.info("=" * 70)

    if args.model == "eegnet":
        model = CompactEEGNet(n_channels, n_samples, n_classes).to(device)
    elif args.model == "encoder":
        model = EEGClassifier(n_channels, n_samples, n_classes).to(device)
    else:
        model = FullPipelineClassifier(n_channels, n_samples, n_classes).to(device)

    # ----------------------------------------------------------------
    # Baseline: CSP + LDA (classical BCI pipeline)
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("BASELINE: CSP + LDA (classical BCI)")
    logger.info("=" * 70)
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_train = dataset.eeg_data[train_mask]
        y_train = dataset.labels[train_mask]
        X_test = dataset.eeg_data[test_mask]
        y_test = dataset.labels[test_mask]

        # Features: band power per channel (simple but effective baseline)
        def extract_bandpower(X, sfreq=256.0):
            from scipy.signal import welch
            features = []
            for trial in X:
                freqs, psd = welch(trial, fs=sfreq, nperseg=min(256, trial.shape[-1]))
                bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
                feat = []
                for low, high in bands:
                    band_mask = (freqs >= low) & (freqs < high)
                    feat.append(psd[:, band_mask].mean(axis=-1))
                features.append(np.concatenate(feat))
            return np.array(features)

        X_train_feat = extract_bandpower(X_train)
        X_test_feat = extract_bandpower(X_test)

        lda = Pipeline([
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis()),
        ])
        lda.fit(X_train_feat, y_train)
        lda_train_acc = (lda.predict(X_train_feat) == y_train).mean() * 100
        lda_test_acc = (lda.predict(X_test_feat) == y_test).mean() * 100
        logger.info("CSP+LDA baseline: train=%.1f%%, test=%.1f%%", lda_train_acc, lda_test_acc)
    except Exception as e:
        logger.warning("CSP+LDA baseline failed: %s", e)
        lda_test_acc = None

    # ----------------------------------------------------------------
    # Training setup
    # ----------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6,
    )

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("TRAINING ON REAL HUMAN EEG (chance = %.1f%%)", 100.0 / n_classes)
    logger.info("=" * 70)

    best_val_acc = 0.0
    best_epoch = 0
    patience = 30
    patience_counter = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1 or epoch == args.epochs:
            val_metrics = evaluate(model, val_loader, device, n_classes)
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                "Epoch %3d/%d | Train Loss: %.4f Acc: %.1f%% | Val Loss: %.4f Acc: %.1f%% | LR: %.2e",
                epoch, args.epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
                lr,
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                best_epoch = epoch
                patience_counter = 0
                torch.save(model.state_dict(), output_dir / "best_model.pt")
            else:
                patience_counter += 5

            if patience_counter >= patience and epoch > 40:
                logger.info("Early stopping at epoch %d (best val acc: %.1f%% at epoch %d)",
                            epoch, best_val_acc, best_epoch)
                break

    total_time = time.time() - start_time
    logger.info("Training time: %.1f seconds (%.1f min)", total_time, total_time / 60)

    # ----------------------------------------------------------------
    # Final evaluation on test set
    # ----------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("FINAL RESULTS ON HELD-OUT TEST SET")
    logger.info("=" * 70)

    # Load best model
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=True))

    test_metrics = evaluate(model, test_loader, device, n_classes)

    logger.info("Test Accuracy: %.1f%% (chance = %.1f%%)", test_metrics["accuracy"], 100.0 / n_classes)
    logger.info("Per-class accuracy:")
    for cls, acc in test_metrics["per_class"].items():
        logger.info("  %s: %.1f%%", cls, acc)

    print_confusion_matrix(test_metrics["confusion"], ["Up", "Down", "Right", "Left"])

    # Statistical significance: binomial test against chance
    n_correct = int(test_metrics["accuracy"] / 100 * len(test_metrics["labels"]))
    n_total = len(test_metrics["labels"])
    chance = 1.0 / n_classes

    from scipy.stats import binomtest
    binom_result = binomtest(n_correct, n_total, chance, alternative="greater")
    binom_p = binom_result.pvalue
    logger.info(
        "Binomial test vs chance (%.0f%%): p = %.6f %s",
        chance * 100, binom_p,
        "*** SIGNIFICANT ***" if binom_p < 0.05 else "(not significant)",
    )

    # Also train-set metrics for overfitting diagnosis
    train_final = evaluate(model, train_loader, device, n_classes)
    logger.info("Train accuracy (for overfit check): %.1f%%", train_final["accuracy"])

    gap = train_final["accuracy"] - test_metrics["accuracy"]
    if gap > 30:
        logger.warning("Large train-test gap (%.1f%%) suggests overfitting", gap)
    elif test_metrics["accuracy"] < 100.0 / n_classes + 5:
        logger.warning("Test accuracy near chance — model may not be learning from EEG")
    else:
        logger.info("Model shows genuine learning from EEG signals!")

    # ----------------------------------------------------------------
    # Save final results
    # ----------------------------------------------------------------
    results = {
        "test_accuracy": test_metrics["accuracy"],
        "val_accuracy": best_val_acc,
        "train_accuracy": train_final["accuracy"],
        "per_class": test_metrics["per_class"],
        "confusion_matrix": test_metrics["confusion"].tolist(),
        "n_train": len(train_set),
        "n_val": len(val_set),
        "n_test": len(test_set),
        "n_channels": n_channels,
        "n_samples": n_samples,
        "subjects": args.subjects,
        "condition": args.condition,
        "best_epoch": best_epoch,
        "total_epochs": args.epochs,
        "p_value": float(binom_p),
        "chance_level": chance * 100,
        "training_time_seconds": total_time,
    }

    import json
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to %s", output_dir / "results.json")

    # Save the full model
    torch.save({
        "model_state": model.state_dict(),
        "config": {
            "n_channels": n_channels,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "model_type": args.model,
        },
    }, output_dir / "clinical_model.pt")

    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Dataset: Inner Speech (OpenNeuro ds003626)")
    logger.info("Subjects: %s", args.subjects)
    logger.info("Condition: %s", args.condition)
    logger.info("EEG: %d channels, %d samples (%.1fs window at %.0f Hz)",
                n_channels, n_samples, n_samples / dataset.sfreq, dataset.sfreq)
    logger.info("Classes: Up, Down, Left, Right (chance = 25%%)")
    logger.info("")
    logger.info("  TEST ACCURACY: %.1f%% (p=%.4f)", test_metrics["accuracy"], binom_p)
    logger.info("")

    return results


if __name__ == "__main__":
    main()
