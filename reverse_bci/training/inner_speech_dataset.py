"""
Inner Speech Dataset loader.

Loads the "Thinking Out Loud" Inner Speech dataset (OpenNeuro ds003626):
- 10 subjects, 128 EEG channels (BioSemi ActiveTwo), 256 Hz
- 4 imagined words: Up, Down, Left, Right (Spanish: Arriba, Abajo, Izquierda, Derecha)
- 3 conditions: Pronounced, Inner Speech, Visualized
- Preprocessed .fif epoch files + pickled event labels

Reference: Nieto et al. (2022), "Thinking out loud, an open-access EEG-based
BCI dataset for inner speech recognition," Scientific Data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional
import pickle
import logging

logger = logging.getLogger(__name__)

DIRECTION_LABELS = {0: "up", 1: "down", 2: "right", 3: "left"}
CONDITION_LABELS = {0: "pronounced", 1: "inner_speech", 2: "visualized"}


class InnerSpeechDataset(Dataset):
    """Loads Inner Speech data from preprocessed .fif files.

    Args:
        data_dir: Root directory containing derivatives/sub-XX/ses-XX/ structure
        subjects: List of subject numbers to include (e.g., [1, 2, 3])
        sessions: List of session numbers (default: [1, 2, 3])
        condition: Which condition to use: "inner_speech", "pronounced", "visualized", or "all"
        t_start: Start of epoch window in seconds relative to stimulus onset (default: 0.5)
        t_end: End of epoch window in seconds relative to stimulus onset (default: 3.5)
        channel_select: Number of channels to select, or None for all 128.
        channel_method: How to select channels:
            "speech" — neuroscience-informed selection targeting Broca's area,
                       Wernicke's area, motor cortex, and supplementary motor area.
            "even"   — evenly spaced across the array (legacy behavior).
        normalize: Whether to z-score normalize per channel per trial
        augment: Whether to apply EEG augmentation during training
    """

    SPEECH_CHANNEL_TIERS = {
        4: ["Fz", "Cz", "C3", "C4"],
        8: ["F7", "F8", "Cz", "C3", "C4", "T7", "T8", "Pz"],
        16: [
            "F7", "F8", "FC5", "FC6", "FC1", "FC2", "FCz",
            "C3", "Cz", "C4", "T7", "T8", "CP5", "CP1", "P3", "Pz",
        ],
        32: [
            "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
            "FC5", "FC1", "FCz", "FC2", "FC6",
            "T7", "C3", "Cz", "C4", "T8",
            "CP5", "CP1", "CPz", "CP2", "CP6",
            "P7", "P3", "Pz", "P4", "P8",
            "O1", "Oz", "O2", "TP7", "TP8",
        ],
        64: [
            "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8",
            "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
            "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
            "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
            "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
            "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
            "PO7", "PO3", "POz", "PO4", "PO8",
            "O1", "Oz", "O2", "Iz",
        ],
    }

    def __init__(
        self,
        data_dir: str,
        subjects: list[int] = None,
        sessions: list[int] = None,
        condition: str = "inner_speech",
        t_start: float = 0.5,
        t_end: float = 3.5,
        channel_select: Optional[int] = None,
        channel_method: str = "speech",
        normalize: bool = True,
        augment: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.subjects = subjects or [1]
        self.sessions = sessions or [1, 2, 3]
        self.condition = condition
        self.t_start = t_start
        self.t_end = t_end
        self.channel_select = channel_select
        self.channel_method = channel_method
        self.normalize = normalize
        self.augment = augment

        self.eeg_data = []
        self.labels = []
        self.subject_ids = []
        self.session_ids = []
        self.sfreq = 256.0
        self.ch_names = None
        self.channel_indices = None

        self._load_all()

    def _load_all(self):
        import mne
        mne.set_log_level("WARNING")

        condition_map = {"inner_speech": 1, "pronounced": 0, "visualized": 2, "all": None}
        target_condition = condition_map.get(self.condition)

        for sub_num in self.subjects:
            for ses_num in self.sessions:
                sub_id = f"sub-{sub_num:02d}"
                ses_id = f"ses-{ses_num:02d}"
                eeg_path = self.data_dir / "derivatives" / sub_id / ses_id / f"{sub_id}_{ses_id}_eeg-epo.fif"
                evt_path = self.data_dir / "derivatives" / sub_id / ses_id / f"{sub_id}_{ses_id}_events.dat"

                if not eeg_path.exists():
                    logger.warning("Missing: %s", eeg_path)
                    continue
                if not evt_path.exists():
                    logger.warning("Missing: %s", evt_path)
                    continue

                epochs = mne.read_epochs(str(eeg_path), verbose="WARNING")
                with open(evt_path, "rb") as f:
                    events = pickle.load(f)

                if self.ch_names is None:
                    self.ch_names = epochs.ch_names
                    self.sfreq = epochs.info["sfreq"]
                    self._setup_channel_selection(epochs.ch_names, epochs)

                data = epochs.get_data()  # (n_trials, n_channels, n_timepoints)
                tmin = epochs.tmin

                t_start_idx = int((self.t_start - tmin) * self.sfreq)
                t_end_idx = int((self.t_end - tmin) * self.sfreq)
                t_start_idx = max(0, t_start_idx)
                t_end_idx = min(data.shape[2], t_end_idx)
                data = data[:, :, t_start_idx:t_end_idx]

                if self.channel_indices is not None:
                    data = data[:, self.channel_indices, :]

                # Filter by condition
                # events columns: [timestamp, direction, condition, session]
                if target_condition is not None:
                    mask = events[:, 2] == target_condition
                else:
                    mask = np.ones(len(events), dtype=bool)

                data = data[mask]
                direction_labels = events[mask, 1]

                if self.normalize:
                    mean = data.mean(axis=-1, keepdims=True)
                    std = data.std(axis=-1, keepdims=True)
                    std = np.where(std < 1e-8, 1.0, std)
                    data = (data - mean) / std

                self.eeg_data.append(data.astype(np.float32))
                self.labels.append(direction_labels.astype(np.int64))
                self.subject_ids.extend([sub_num] * len(direction_labels))
                self.session_ids.extend([ses_num] * len(direction_labels))

                logger.info(
                    "Loaded %s %s: %d trials (%s condition), %d channels, %d samples",
                    sub_id, ses_id, len(direction_labels), self.condition,
                    data.shape[1], data.shape[2],
                )

        if not self.eeg_data:
            raise FileNotFoundError(f"No data found in {self.data_dir}")

        self.eeg_data = np.concatenate(self.eeg_data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        self.subject_ids = np.array(self.subject_ids)
        self.session_ids = np.array(self.session_ids)

        logger.info(
            "Dataset ready: %d trials, %d channels, %d samples, %d classes",
            len(self.labels), self.eeg_data.shape[1], self.eeg_data.shape[2],
            len(np.unique(self.labels)),
        )
        for d, name in DIRECTION_LABELS.items():
            count = (self.labels == d).sum()
            logger.info("  %s: %d trials", name, count)

    def _setup_channel_selection(self, ch_names: list[str], epochs=None):
        if self.channel_select is None:
            self.channel_indices = None
            return

        n_select = min(self.channel_select, len(ch_names))

        if n_select == len(ch_names):
            self.channel_indices = None
            return

        if self.channel_method == "speech":
            self.channel_indices = self._select_speech_channels(
                ch_names, n_select, epochs,
            )
        else:
            self.channel_indices = np.linspace(
                0, len(ch_names) - 1, n_select, dtype=int,
            )

        selected_names = [ch_names[i] for i in self.channel_indices]
        logger.info(
            "Selected %d/%d channels (%s): %s",
            len(self.channel_indices), len(ch_names),
            self.channel_method, selected_names,
        )

    # BioSemi 128 speech-area channel mapping.
    # The exact mapping is loaded from proprietary_config.py (gitignored).
    # Public fallback uses evenly-spaced channels.
    try:
        from reverse_bci.proprietary_config import (
            BIOSEMI128_SPEECH_TIERS as _PROP_TIERS,
            BIOSEMI128_SPEECH_REGIONS as _PROP_REGIONS,
        )
        BIOSEMI128_SPEECH_REGIONS = _PROP_REGIONS
        BIOSEMI128_SPEECH_TIERS = _PROP_TIERS
    except ImportError:
        BIOSEMI128_SPEECH_REGIONS = {}
        BIOSEMI128_SPEECH_TIERS = {}

    def _select_speech_channels(
        self, ch_names: list[str], n_select: int, epochs=None,
    ) -> np.ndarray:
        """Select channels targeting speech/motor brain regions.

        Strategy:
        1. Look up the nearest predefined tier (4/8/16/32/64 channels).
        2. Try direct name matching against the epoch's channel list
           (works when channels use standard 10-20 names).
        3. If BioSemi naming detected (A1-D32), use the hardcoded
           BioSemi-128 speech-area mapping.
        4. Fall back to MNE montage 3-D position nearest-neighbor.
        5. Final fallback: evenly spaced.
        """
        import mne

        tiers = sorted(self.SPEECH_CHANNEL_TIERS.keys())
        tier_key = tiers[-1]
        for t in tiers:
            if t >= n_select:
                tier_key = t
                break
        target_names = self.SPEECH_CHANNEL_TIERS[tier_key][:n_select]

        # Strategy 1: direct 10-20 name matching
        ch_upper = {name.upper(): i for i, name in enumerate(ch_names)}
        direct = [ch_upper[t.upper()] for t in target_names
                  if t.upper() in ch_upper]
        if len(direct) >= n_select:
            return np.array(sorted(direct[:n_select]))

        # Strategy 2: BioSemi A1-D32 naming detected
        is_biosemi = any(ch.startswith(("A", "B", "C", "D"))
                         and ch[1:].isdigit() for ch in ch_names[:5])
        if is_biosemi:
            bio_tiers = sorted(self.BIOSEMI128_SPEECH_TIERS.keys())
            bio_key = bio_tiers[-1]
            for bt in bio_tiers:
                if bt >= n_select:
                    bio_key = bt
                    break
            bio_targets = self.BIOSEMI128_SPEECH_TIERS[bio_key][:n_select]
            bio_idx = [ch_upper[t.upper()] for t in bio_targets
                       if t.upper() in ch_upper]
            if len(bio_idx) >= n_select:
                logger.info("Using BioSemi-128 speech-area mapping")
                return np.array(sorted(bio_idx[:n_select]))
            # Partial match — fill remaining from all speech regions
            all_speech = []
            for region_chs in self.BIOSEMI128_SPEECH_REGIONS.values():
                for ch in region_chs:
                    if ch.upper() in ch_upper and ch_upper[ch.upper()] not in bio_idx:
                        all_speech.append(ch_upper[ch.upper()])
            bio_idx.extend(all_speech)
            if len(bio_idx) >= n_select:
                logger.info("Using BioSemi-128 speech-area mapping (extended)")
                return np.array(sorted(list(dict.fromkeys(bio_idx))[:n_select]))

        # Strategy 3: MNE montage 3-D position nearest-neighbor
        if epochs is not None:
            try:
                montage = epochs.get_montage()
                if montage is not None:
                    return self._nearest_by_position(
                        ch_names, target_names, montage, n_select,
                    )
            except Exception:
                pass

        # Strategy 4: evenly spaced
        return np.linspace(0, len(ch_names) - 1, n_select, dtype=int)

    @staticmethod
    def _nearest_by_position(
        ch_names: list[str],
        target_names: list[str],
        montage,
        n_select: int,
    ) -> np.ndarray:
        """Pick data channels closest to target 10-20 positions."""
        import mne

        pos = montage.get_positions()["ch_pos"]
        std = mne.channels.make_standard_montage("standard_1020")
        std_pos = std.get_positions()["ch_pos"]

        selected: list[int] = []
        used: set[int] = set()
        for target in target_names:
            if target not in std_pos:
                continue
            tgt = np.array(std_pos[target])
            best_idx, best_dist = 0, float("inf")
            for i, ch in enumerate(ch_names):
                if i in used or ch not in pos:
                    continue
                d = np.linalg.norm(np.array(pos[ch]) - tgt)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            selected.append(best_idx)
            used.add(best_idx)
            if len(selected) >= n_select:
                break

        if len(selected) < n_select:
            remaining = [i for i in range(len(ch_names)) if i not in used]
            selected.extend(remaining[: n_select - len(selected)])

        return np.array(sorted(selected[:n_select]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.augment:
            eeg = self._augment(eeg)

        return {"eeg": eeg, "label": label}

    def _augment(self, eeg: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        eeg = eeg + torch.randn_like(eeg) * 0.05

        # Random time shift (up to 25ms)
        shift = torch.randint(-6, 7, (1,)).item()
        if shift != 0:
            eeg = torch.roll(eeg, shift, dims=-1)

        # Channel dropout (5%)
        mask = torch.rand(eeg.shape[0]) > 0.05
        eeg = eeg * mask.unsqueeze(-1).float()

        # Amplitude jitter
        scale = 0.9 + torch.rand(1).item() * 0.2
        eeg = eeg * scale

        return eeg

    def get_split(self, train_ratio: float = 0.7, val_ratio: float = 0.15, seed: int = 42,
                  mode: str = "stratified"):
        """Split dataset into train/val/test.

        Modes:
            "stratified": Stratified random split within each session (default, best for small data)
            "cross_session": Session 1+2 train, session 3 val+test (harder, tests generalization)
        """
        rng = np.random.RandomState(seed)

        if mode == "cross_session":
            unique_sessions = np.unique(self.session_ids)
            if len(unique_sessions) >= 3:
                train_mask = np.isin(self.session_ids, [unique_sessions[0], unique_sessions[1]])
                remaining = ~train_mask
                remaining_idx = np.where(remaining)[0]
                rng.shuffle(remaining_idx)
                n_val = len(remaining_idx) // 2
                val_idx = remaining_idx[:n_val]
                test_idx = remaining_idx[n_val:]

                val_mask = np.zeros(len(self.labels), dtype=bool)
                test_mask = np.zeros(len(self.labels), dtype=bool)
                val_mask[val_idx] = True
                test_mask[test_idx] = True
                return train_mask, val_mask, test_mask

        # Stratified split: preserves class balance in each fold
        n = len(self.labels)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        for label in np.unique(self.labels):
            idx = np.where(self.labels == label)[0]
            rng.shuffle(idx)
            n_train = int(len(idx) * train_ratio)
            n_val = int(len(idx) * val_ratio)
            train_mask[idx[:n_train]] = True
            val_mask[idx[n_train:n_train + n_val]] = True
            test_mask[idx[n_train + n_val:]] = True

        return train_mask, val_mask, test_mask

    def subset(self, mask: np.ndarray) -> "InnerSpeechSubset":
        return InnerSpeechSubset(
            self.eeg_data[mask],
            self.labels[mask],
            augment=self.augment,
        )


class InnerSpeechSubset(Dataset):
    def __init__(self, eeg_data: np.ndarray, labels: np.ndarray, augment: bool = False):
        self.eeg_data = eeg_data
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        eeg = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.augment:
            eeg = eeg + torch.randn_like(eeg) * 0.05
            shift = torch.randint(-6, 7, (1,)).item()
            if shift != 0:
                eeg = torch.roll(eeg, shift, dims=-1)
            mask = torch.rand(eeg.shape[0]) > 0.05
            eeg = eeg * mask.unsqueeze(-1).float()
            scale = 0.9 + torch.rand(1).item() * 0.2
            eeg = eeg * scale

        return {"eeg": eeg, "label": label}
