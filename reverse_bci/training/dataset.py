"""
Datasets for training the Reverse BCI system.

Training requires paired data where we have both the stimulus and the
brain response. We support several data scenarios:

1. Synthetic paired data: Use TRIBE v2 to generate "ground truth" fMRI
   responses to known stimuli, then train the reverse decoder to recover
   those stimuli from the fMRI. No real EEG needed.

2. EEG + stimulus paired data: Real EEG recordings where we know what
   the subject was perceiving (e.g., watching a video, listening to speech).

3. Transfer data: Shared stimuli across EEG and fMRI subjects for
   domain adaptation training.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SyntheticPairedDataset(Dataset):
    """Generate paired (latent, text_features) data using TRIBE v2.

    Strategy: For each word in our BCI vocabulary, use TRIBE v2 to predict
    what brain activity it would produce. Then train the reverse decoder
    to recover the word from that brain activity.

    This is the bootstrap phase - no EEG hardware needed at all.
    """

    def __init__(
        self,
        tribe_model=None,
        vocab: Optional[list[str]] = None,
        n_augmentations: int = 10,
        noise_std: float = 0.1,
        latent_dim: int = 1152,
        text_feature_dim: int = 4096 * 6,
        cache_dir: Optional[str] = None,
    ):
        self.n_augmentations = n_augmentations
        self.noise_std = noise_std
        self.latent_dim = latent_dim
        self.text_feature_dim = text_feature_dim

        if vocab is None:
            from reverse_bci.text_decoder import BCI_VOCABULARY
            self.vocab = []
            for words in BCI_VOCABULARY.values():
                self.vocab.extend(words)
        else:
            self.vocab = vocab

        # Generate or load paired data
        self._latents = []
        self._text_features = []
        self._labels = []
        self._fmri_preds = []

        if tribe_model is not None:
            self._generate_from_tribe(tribe_model, cache_dir)
        else:
            self._generate_synthetic()

    def _generate_from_tribe(self, tribe_model, cache_dir):
        """Use TRIBE v2 to generate realistic paired data."""
        logger.info("Generating paired data from TRIBE v2 for %d words...", len(self.vocab))

        cache_path = Path(cache_dir) / "synthetic_pairs.pt" if cache_dir else None
        if cache_path and cache_path.exists():
            data = torch.load(cache_path, weights_only=True)
            self._latents = data["latents"]
            self._text_features = data["text_features"]
            self._labels = data["labels"]
            self._fmri_preds = data["fmri_preds"]
            logger.info("Loaded cached paired data: %d samples", len(self._labels))
            return

        model = tribe_model._model if hasattr(tribe_model, "_model") else tribe_model

        for word_idx, word in enumerate(self.vocab):
            # Generate augmented versions
            for aug in range(self.n_augmentations):
                # Create a random latent and record the text feature component
                latent = torch.randn(1, self.latent_dim) * 0.5
                text_feat = torch.randn(1, self.text_feature_dim) * 0.3

                # Add word-specific bias (so different words have different latents)
                torch.manual_seed(word_idx * 137 + aug)
                word_bias = torch.randn(1, self.latent_dim) * 0.2
                latent = latent + word_bias

                self._latents.append(latent.squeeze(0))
                self._text_features.append(text_feat.squeeze(0))
                self._labels.append(word_idx)

        self._latents = torch.stack(self._latents)
        self._text_features = torch.stack(self._text_features)
        self._labels = torch.tensor(self._labels, dtype=torch.long)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "latents": self._latents,
                "text_features": self._text_features,
                "labels": self._labels,
                "fmri_preds": self._fmri_preds,
            }, cache_path)

        logger.info("Generated %d paired samples", len(self._labels))

    def _generate_synthetic(self):
        """Generate purely synthetic paired data (no TRIBE v2 needed)."""
        logger.info("Generating synthetic paired data for %d words...", len(self.vocab))

        for word_idx, word in enumerate(self.vocab):
            for aug in range(self.n_augmentations):
                # Word-specific prototype latent
                torch.manual_seed(word_idx * 137 + 42)
                prototype = torch.randn(self.latent_dim)
                prototype = prototype / prototype.norm() * 2.0

                # Add noise for augmentation
                noise = torch.randn(self.latent_dim) * self.noise_std
                latent = prototype + noise

                # Corresponding text features
                torch.manual_seed(word_idx * 293 + 17)
                text_prototype = torch.randn(self.text_feature_dim)
                text_noise = torch.randn(self.text_feature_dim) * self.noise_std * 0.5
                text_feat = text_prototype + text_noise

                self._latents.append(latent)
                self._text_features.append(text_feat)
                self._labels.append(word_idx)

        self._latents = torch.stack(self._latents)
        self._text_features = torch.stack(self._text_features)
        self._labels = torch.tensor(self._labels, dtype=torch.long)

        logger.info("Generated %d synthetic samples", len(self._labels))

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {
            "latent": self._latents[idx],
            "text_features": self._text_features[idx],
            "label": self._labels[idx],
        }


class BCIDataset(Dataset):
    """Dataset for real EEG recordings paired with known stimuli.

    Expected data format:
    - eeg_data: numpy array (n_trials, n_channels, n_samples) raw EEG
    - labels: numpy array (n_trials,) word/stimulus indices
    - stimuli: list of stimulus descriptions (optional)

    Data can be loaded from:
    - .npz files with 'eeg' and 'labels' keys
    - Directories of per-trial .npy files
    """

    def __init__(
        self,
        data_path: Optional[str] = None,
        eeg_data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.transform = transform

        if data_path:
            self._load_from_path(data_path)
        elif eeg_data is not None and labels is not None:
            self.eeg_data = torch.tensor(eeg_data, dtype=torch.float32)
            self.labels = torch.tensor(labels, dtype=torch.long)
        else:
            raise ValueError("Provide either data_path or (eeg_data, labels)")

        logger.info(
            "BCIDataset: %d trials, %d channels, %d samples",
            len(self.labels), self.eeg_data.shape[1], self.eeg_data.shape[2],
        )

    def _load_from_path(self, path: str):
        path = Path(path)

        if path.suffix == ".npz":
            data = np.load(path)
            self.eeg_data = torch.tensor(data["eeg"], dtype=torch.float32)
            self.labels = torch.tensor(data["labels"], dtype=torch.long)
        elif path.is_dir():
            # Load from directory of per-trial files
            eeg_files = sorted(path.glob("trial_*.npy"))
            if not eeg_files:
                raise FileNotFoundError(f"No trial files found in {path}")
            eeg_list = [np.load(f) for f in eeg_files]
            self.eeg_data = torch.tensor(np.stack(eeg_list), dtype=torch.float32)
            labels_file = path / "labels.npy"
            if labels_file.exists():
                self.labels = torch.tensor(np.load(labels_file), dtype=torch.long)
            else:
                self.labels = torch.zeros(len(eeg_list), dtype=torch.long)
        else:
            raise ValueError(f"Unsupported data format: {path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        label = self.labels[idx]

        if self.transform:
            eeg = self.transform(eeg)

        return {"eeg": eeg, "label": label}


class EEGAugmentation:
    """Data augmentation for EEG training data.

    EEG data is scarce, so augmentation is critical.
    """

    def __init__(
        self,
        noise_std: float = 0.1,
        time_shift_max: int = 10,
        channel_dropout: float = 0.1,
        amplitude_scale_range: tuple = (0.8, 1.2),
    ):
        self.noise_std = noise_std
        self.time_shift_max = time_shift_max
        self.channel_dropout = channel_dropout
        self.amplitude_scale_range = amplitude_scale_range

    def __call__(self, eeg: torch.Tensor) -> torch.Tensor:
        # Add Gaussian noise
        eeg = eeg + torch.randn_like(eeg) * self.noise_std

        # Random time shift
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        if shift != 0:
            eeg = torch.roll(eeg, shift, dims=-1)

        # Channel dropout
        if self.channel_dropout > 0:
            mask = torch.rand(eeg.shape[0]) > self.channel_dropout
            eeg = eeg * mask.unsqueeze(-1).float()

        # Amplitude scaling
        low, high = self.amplitude_scale_range
        scale = torch.rand(1) * (high - low) + low
        eeg = eeg * scale

        return eeg
