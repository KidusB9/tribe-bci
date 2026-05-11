"""
EEG Signal Processing Pipeline - Production Grade.

The #1 failure mode of consumer BCI: the model learns to decode muscle
artifacts (EMG), not brain signals. A jaw clench is 10-100x stronger
than a brainwave. If we don't strip EMG/EOG/ECG before training, the
model will cheat on able-bodied users and fail completely on paralyzed
users (who can't produce those artifacts).

Pipeline:
    Raw EEG -> Bandpass (0.5-45 Hz) -> Notch (50/60 Hz)
    -> ICA artifact rejection (EOG, EMG, ECG component removal)
    -> Amplitude-based epoch rejection
    -> Re-referencing (CAR)
    -> Session normalization (combat biological drift)
    -> Feature Extraction (band powers, phase, connectivity)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class EEGBand(Enum):
    DELTA = (0.5, 4.0)
    THETA = (4.0, 8.0)
    ALPHA = (8.0, 13.0)
    BETA = (13.0, 30.0)
    GAMMA = (30.0, 45.0)


HEADSET_MONTAGES = {
    "muse": {
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "sfreq": 256,
        "n_channels": 4,
        "frontal_idx": [1, 2],       # AF7, AF8 - blink artifacts live here
        "temporal_idx": [0, 3],       # TP9, TP10 - muscle artifacts live here
    },
    "muse_s": {
        "channels": ["TP9", "AF7", "AF8", "TP10"],
        "sfreq": 256,
        "n_channels": 4,
        "frontal_idx": [1, 2],
        "temporal_idx": [0, 3],
    },
    "emotiv_epoc": {
        "channels": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        ],
        "sfreq": 128,
        "n_channels": 14,
        "frontal_idx": [0, 1, 2, 11, 12, 13],
        "temporal_idx": [4, 9],
    },
    "emotiv_epoc_x": {
        "channels": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        ],
        "sfreq": 256,
        "n_channels": 14,
        "frontal_idx": [0, 1, 2, 11, 12, 13],
        "temporal_idx": [4, 9],
    },
    "emotiv_insight": {
        "channels": ["AF3", "AF4", "T7", "T8", "Pz"],
        "sfreq": 128,
        "n_channels": 5,
        "frontal_idx": [0, 1],
        "temporal_idx": [2, 3],
    },
    "openbci_cyton": {
        "channels": ["Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2"],
        "sfreq": 250,
        "n_channels": 8,
        "frontal_idx": [0, 1],
        "temporal_idx": [],
    },
    "openbci_daisy": {
        "channels": [
            "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2",
            "F7", "F8", "F3", "F4", "T7", "T8", "P3", "P4",
        ],
        "sfreq": 125,
        "n_channels": 16,
        "frontal_idx": [0, 1, 8, 9, 10, 11],
        "temporal_idx": [12, 13],
    },
}


# Frequency bands diagnostic for artifact detection
# EMG: high-frequency power > 20 Hz, broadband
# EOG: very low frequency < 4 Hz, huge amplitude on frontal channels
# ECG: ~1 Hz periodic, sharp QRS complex
ARTIFACT_SIGNATURES = {
    "emg": {"freq_range": (20, 45), "spatial": "temporal", "amplitude_ratio": 3.0},
    "eog": {"freq_range": (0.5, 4), "spatial": "frontal", "amplitude_ratio": 5.0},
    "ecg": {"freq_range": (0.5, 3), "spatial": "all", "periodicity_hz": (0.8, 2.0)},
}


@dataclass
class EEGProcessorConfig:
    headset: str = "muse"
    bandpass_low: float = 0.5
    bandpass_high: float = 45.0
    notch_freq: float = 60.0
    notch_width: float = 2.0
    artifact_threshold_uv: float = 100.0
    window_seconds: float = 2.0
    overlap_seconds: float = 1.0
    apply_car: bool = True
    apply_surface_laplacian: bool = True
    compute_band_powers: bool = True
    compute_phase: bool = True
    compute_connectivity: bool = True
    # ICA artifact rejection
    use_ica: bool = True
    ica_calibration_seconds: float = 60.0
    emg_rejection: bool = True
    eog_rejection: bool = True
    ecg_rejection: bool = True
    # Session normalization (combat biological drift)
    session_normalize: bool = True
    # Strict mode: reject windows with ANY detected artifact rather than interpolating
    strict_artifact_rejection: bool = False


class ArtifactDetector:
    """Detects and classifies EEG artifacts by type.

    This is the make-or-break component. If EMG leaks through, the
    decoder learns to read muscle twitches, not thoughts. On a paralyzed
    user, that means 0% accuracy.
    """

    def __init__(self, n_channels: int, sfreq: float, montage: dict):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.frontal_idx = montage.get("frontal_idx", [])
        self.temporal_idx = montage.get("temporal_idx", [])
        self._calibrated = False
        self._baseline_band_power = None
        self._baseline_amplitude = None

    def calibrate(self, data: np.ndarray):
        """Compute baseline statistics from clean resting-state EEG.

        Call this during calibration phase with eyes-open resting data.
        """
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, 1.0 / self.sfreq)

        self._baseline_band_power = {}
        for name, sig in ARTIFACT_SIGNATURES.items():
            low, high = sig["freq_range"]
            mask = (freqs >= low) & (freqs <= high)
            powers = []
            for ch in range(self.n_channels):
                spectrum = np.abs(np.fft.rfft(data[ch])) ** 2 / n
                powers.append(spectrum[mask].mean())
            self._baseline_band_power[name] = np.array(powers)

        self._baseline_amplitude = np.std(data, axis=1)
        self._calibrated = True
        logger.info(
            "ArtifactDetector calibrated: baseline amplitude=%s",
            [f"{a:.1f}" for a in self._baseline_amplitude],
        )

    def detect(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Detect artifacts in a window of EEG data.

        Returns dict mapping artifact type to boolean mask (n_samples,)
        indicating contaminated time points.
        """
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, 1.0 / self.sfreq)
        artifacts = {}

        # --- EOG detection (eye blinks) ---
        eog_mask = np.zeros(n, dtype=bool)
        if self.frontal_idx:
            frontal = data[self.frontal_idx]
            # Blinks: large slow deflections on frontal channels
            # Detect via amplitude envelope in delta band (0.5-4 Hz)
            for ch_data in frontal:
                spectrum = np.fft.rfft(ch_data)
                delta_mask = (freqs >= 0.5) & (freqs <= 4.0)
                filtered = np.zeros_like(spectrum)
                filtered[delta_mask] = spectrum[delta_mask]
                delta_signal = np.fft.irfft(filtered, n=n)
                envelope = np.abs(delta_signal)

                # Threshold: 3x median or calibrated baseline
                if self._calibrated:
                    thresh = self._baseline_amplitude[self.frontal_idx].mean() * 3.0
                else:
                    thresh = np.median(envelope) * 5.0
                blink_samples = envelope > thresh

                # Expand blink detection window (blinks last ~200-400ms)
                expand = int(0.2 * self.sfreq)
                for idx in np.where(blink_samples)[0]:
                    start = max(0, idx - expand)
                    end = min(n, idx + expand)
                    eog_mask[start:end] = True

        artifacts["eog"] = eog_mask

        # --- EMG detection (muscle activity) ---
        emg_mask = np.zeros(n, dtype=bool)
        # EMG has broadband high-frequency power (>20 Hz)
        hf_mask = (freqs >= 20) & (freqs <= 45)
        lf_mask = (freqs >= 4) & (freqs <= 13)

        for ch in range(self.n_channels):
            spectrum = np.abs(np.fft.rfft(data[ch])) ** 2 / n
            hf_power = spectrum[hf_mask].mean() if hf_mask.any() else 0
            lf_power = spectrum[lf_mask].mean() if lf_mask.any() else 1e-10

            # EMG ratio: high-frequency to low-frequency power
            emg_ratio = hf_power / max(lf_power, 1e-10)

            if self._calibrated:
                baseline_hf = self._baseline_band_power["emg"][ch]
                baseline_lf = max(self._baseline_band_power["eog"][ch], 1e-10)
                baseline_ratio = baseline_hf / baseline_lf
                threshold = baseline_ratio * ARTIFACT_SIGNATURES["emg"]["amplitude_ratio"]
            else:
                threshold = ARTIFACT_SIGNATURES["emg"]["amplitude_ratio"]

            if emg_ratio > threshold:
                # This channel is EMG-contaminated for the whole window
                # Use sliding RMS to find the exact contaminated segments
                window_len = int(0.1 * self.sfreq)  # 100ms RMS window
                if window_len > 0 and n > window_len:
                    rms = np.sqrt(
                        np.convolve(data[ch] ** 2, np.ones(window_len) / window_len, mode="same")
                    )
                    rms_thresh = np.median(rms) * 3.0
                    emg_mask |= rms > rms_thresh

        artifacts["emg"] = emg_mask

        # --- ECG detection (heartbeat) ---
        ecg_mask = np.zeros(n, dtype=bool)
        # Detect periodic sharp peaks consistent with QRS complex (~1Hz)
        for ch in range(self.n_channels):
            # Look for sharp transients
            gradient = np.abs(np.diff(data[ch], prepend=data[ch, 0]))
            grad_thresh = np.percentile(gradient, 97)
            peaks = gradient > grad_thresh

            # Check if peaks are periodic (0.8-2.0 Hz = 50-75 bpm)
            peak_idx = np.where(peaks)[0]
            if len(peak_idx) >= 3:
                intervals = np.diff(peak_idx) / self.sfreq
                mean_interval = np.median(intervals)
                if 0.5 < mean_interval < 1.25:  # 48-120 bpm
                    interval_std = np.std(intervals)
                    if interval_std < mean_interval * 0.3:
                        # Regular heartbeat-like pattern detected
                        for idx in peak_idx:
                            start = max(0, idx - int(0.05 * self.sfreq))
                            end = min(n, idx + int(0.05 * self.sfreq))
                            ecg_mask[start:end] = True
                        break

        artifacts["ecg"] = ecg_mask

        # Combined mask
        artifacts["any"] = eog_mask | emg_mask | ecg_mask

        return artifacts

    def get_artifact_summary(self, artifacts: dict, window_seconds: float) -> dict:
        """Summarize artifact contamination for quality reporting."""
        n = len(artifacts["any"])
        return {
            "eog_ratio": float(artifacts["eog"].mean()),
            "emg_ratio": float(artifacts["emg"].mean()),
            "ecg_ratio": float(artifacts["ecg"].mean()),
            "total_contaminated_ratio": float(artifacts["any"].mean()),
            "is_clean": float(artifacts["any"].mean()) < 0.3,
            "dominant_artifact": max(
                ["eog", "emg", "ecg"],
                key=lambda k: artifacts[k].mean(),
            ) if artifacts["any"].any() else "none",
        }


class ICADecomposer:
    """Lightweight ICA for consumer EEG artifact removal.

    With only 4-14 channels, full ICA (like in MNE-Python) is overkill
    and numerically unstable. We use a simplified FastICA tailored for
    the known artifact structure of consumer EEG:

    - Component 1-2: likely EOG (frontal channels dominate)
    - Component 3+: likely EMG (high-frequency, temporal channels)

    For proper ICA on 14+ channels, we delegate to MNE-Python.
    """

    def __init__(self, n_channels: int, sfreq: float, montage: dict):
        self.n_channels = n_channels
        self.sfreq = sfreq
        self.frontal_idx = montage.get("frontal_idx", [])
        self.temporal_idx = montage.get("temporal_idx", [])
        self._unmixing = None
        self._mixing = None
        self._component_labels = None
        self._fitted = False
        self._use_mne = n_channels >= 8

    def fit(self, calibration_data: np.ndarray):
        """Fit ICA on calibration data to learn artifact components.

        Args:
            calibration_data: (n_channels, n_samples) of resting-state EEG
        """
        if self._use_mne:
            self._fit_mne(calibration_data)
        else:
            self._fit_simple(calibration_data)
        self._fitted = True

    def _fit_simple(self, data: np.ndarray):
        """Simplified FastICA for low-channel-count headsets (4-5 ch)."""
        n_ch, n_samp = data.shape

        # Center and whiten
        mean = data.mean(axis=1, keepdims=True)
        centered = data - mean
        cov = np.cov(centered)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
        whitened = whitening @ centered

        # FastICA iteration
        n_components = n_ch
        W = np.random.randn(n_components, n_ch) * 0.01
        W, _ = np.linalg.qr(W.T)
        W = W.T

        for iteration in range(200):
            # G(u) = tanh(u), g(u) = 1 - tanh^2(u)
            WX = W @ whitened
            tanh_WX = np.tanh(WX)
            W_new = (tanh_WX @ whitened.T) / n_samp - (
                (1 - tanh_WX ** 2).mean(axis=1, keepdims=True) * W
            )

            # Symmetric decorrelation
            U, S, Vt = np.linalg.svd(W_new, full_matrices=False)
            W_new = U @ Vt

            # Check convergence
            change = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1))
            W = W_new
            if change < 1e-6:
                break

        self._unmixing = W @ whitening
        self._mixing = np.linalg.pinv(self._unmixing)

        # Label components as artifact or neural
        self._label_components(data)

    def _fit_mne(self, data: np.ndarray):
        """Full ICA via MNE-Python for higher-channel-count headsets."""
        try:
            import mne
            from mne.preprocessing import ICA

            info = mne.create_info(
                ch_names=[f"EEG{i}" for i in range(self.n_channels)],
                sfreq=self.sfreq,
                ch_types="eeg",
            )
            raw = mne.io.RawArray(data * 1e-6, info, verbose=False)  # uV -> V
            raw.filter(1.0, 40.0, verbose=False)

            n_components = min(self.n_channels - 1, 15)
            ica = ICA(n_components=n_components, method="fastica", random_state=42)
            ica.fit(raw, verbose=False)

            sources = ica.get_sources(raw).get_data() * 1e6  # back to uV
            self._unmixing = ica.unmixing_matrix_
            self._mixing = ica.mixing_matrix_
            self._mne_ica = ica

            self._label_components_from_sources(sources)
            logger.info("MNE ICA fitted: %d components", n_components)

        except ImportError:
            logger.warning("MNE not available, falling back to simple ICA")
            self._use_mne = False
            self._fit_simple(data)

    def _label_components(self, data: np.ndarray):
        """Label ICA components as artifact or neural based on spatial/spectral properties."""
        sources = self._unmixing @ data
        self._label_components_from_sources(sources)

    def _label_components_from_sources(self, sources: np.ndarray):
        """Label components by analyzing their spatial and spectral signatures."""
        n_comp = sources.shape[0]
        n_samp = sources.shape[1]
        self._component_labels = ["neural"] * n_comp

        freqs = np.fft.rfftfreq(n_samp, 1.0 / self.sfreq)
        hf_mask = (freqs >= 20) & (freqs <= 45)
        lf_mask = (freqs >= 1) & (freqs <= 8)
        delta_mask = (freqs >= 0.5) & (freqs <= 4)

        for i in range(n_comp):
            spectrum = np.abs(np.fft.rfft(sources[i])) ** 2 / n_samp
            hf_power = spectrum[hf_mask].mean() if hf_mask.any() else 0
            lf_power = spectrum[lf_mask].mean() if lf_mask.any() else 1e-10
            delta_power = spectrum[delta_mask].mean() if delta_mask.any() else 0
            total_power = spectrum.mean()

            # EMG: dominated by high-frequency power
            if hf_power / max(total_power, 1e-10) > 0.4:
                self._component_labels[i] = "emg"
                continue

            # EOG: dominated by very low frequency, large amplitude
            if delta_power / max(total_power, 1e-10) > 0.6:
                # Check if spatial loading is frontal
                if self._mixing is not None and self.frontal_idx:
                    frontal_loading = np.abs(self._mixing[self.frontal_idx, i]).mean()
                    other_idx = [j for j in range(self.n_channels) if j not in self.frontal_idx]
                    other_loading = np.abs(self._mixing[other_idx, i]).mean() if other_idx else 1e-10
                    if frontal_loading > other_loading * 1.5:
                        self._component_labels[i] = "eog"
                        continue

            # ECG: periodic sharp transients
            gradient = np.abs(np.diff(sources[i]))
            peak_thresh = np.percentile(gradient, 97)
            peaks = np.where(gradient > peak_thresh)[0]
            if len(peaks) >= 5:
                intervals = np.diff(peaks) / self.sfreq
                if len(intervals) > 0:
                    median_interval = np.median(intervals)
                    if 0.5 < median_interval < 1.25:
                        interval_cv = np.std(intervals) / max(median_interval, 1e-10)
                        if interval_cv < 0.3:
                            self._component_labels[i] = "ecg"

        artifact_count = sum(1 for l in self._component_labels if l != "neural")
        logger.info(
            "ICA component labels: %s (%d artifacts / %d total)",
            self._component_labels, artifact_count, n_comp,
        )

    def remove_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Remove artifact components from EEG data.

        Projects data into ICA space, zeros out artifact components,
        and projects back.
        """
        if not self._fitted:
            return data

        sources = self._unmixing @ data

        for i, label in enumerate(self._component_labels):
            if label != "neural":
                sources[i] = 0.0

        cleaned = self._mixing @ sources
        return cleaned

    @property
    def is_fitted(self) -> bool:
        return self._fitted


class SessionNormalizer:
    """Combats day-to-day biological drift.

    The brain's baseline state changes with sleep, caffeine, stress,
    and even headset placement. A model trained on Monday will fail
    on Tuesday unless we normalize for these session-level shifts.

    Strategy: during calibration, compute per-channel mean and variance.
    During decoding, z-score normalize each window against the session baseline.
    """

    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self._mean = None
        self._std = None
        self._band_baselines = None
        self._calibrated = False

    def calibrate(self, data: np.ndarray, band_powers: Optional[dict] = None):
        """Compute session baseline from calibration data.

        Args:
            data: (n_channels, n_samples) of resting-state EEG (already filtered)
            band_powers: optional dict of baseline band power per channel
        """
        self._mean = data.mean(axis=1)
        self._std = data.std(axis=1)
        self._std = np.maximum(self._std, 1e-6)

        if band_powers is not None:
            self._band_baselines = {
                band: powers.copy() for band, powers in band_powers.items()
            }

        self._calibrated = True
        logger.info(
            "SessionNormalizer calibrated: mean=%s, std=%s",
            [f"{m:.2f}" for m in self._mean],
            [f"{s:.2f}" for s in self._std],
        )

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalize data against session baseline."""
        if not self._calibrated:
            return data
        return (data - self._mean[:, np.newaxis]) / self._std[:, np.newaxis]

    def normalize_band_powers(self, band_powers: dict) -> dict:
        """Normalize band powers relative to session baseline."""
        if not self._calibrated or self._band_baselines is None:
            return band_powers
        normalized = {}
        for band, powers in band_powers.items():
            if band in self._band_baselines:
                baseline = np.maximum(self._band_baselines[band], 1e-6)
                normalized[band] = (powers - baseline) / baseline
            else:
                normalized[band] = powers
        return normalized

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated


class EEGProcessor:
    """Production-grade EEG signal processing pipeline.

    The processing order matters enormously:
    1. Bandpass + notch (remove out-of-band noise)
    2. ICA artifact rejection (remove EOG/EMG/ECG BEFORE any learning)
    3. Amplitude rejection (catch anything ICA missed)
    4. CAR re-referencing
    5. Session normalization (combat biological drift)
    6. Feature extraction
    """

    def __init__(self, config: Optional[EEGProcessorConfig] = None):
        if config is None:
            config = EEGProcessorConfig()
        self.config = config

        montage = HEADSET_MONTAGES.get(config.headset)
        if montage is None:
            raise ValueError(
                f"Unknown headset '{config.headset}'. "
                f"Supported: {list(HEADSET_MONTAGES.keys())}"
            )
        self.channels = montage["channels"]
        self.sfreq = montage["sfreq"]
        self.n_channels = montage["n_channels"]
        self._montage = montage

        self.window_samples = int(config.window_seconds * self.sfreq)
        self.overlap_samples = int(config.overlap_seconds * self.sfreq)

        self._buffer = np.zeros((self.n_channels, 0))
        self._artifact_counts = np.zeros(self.n_channels, dtype=int)
        self._total_windows = 0
        self._rejected_windows = 0

        # Artifact subsystems
        self.artifact_detector = ArtifactDetector(self.n_channels, self.sfreq, montage)
        self.ica = ICADecomposer(self.n_channels, self.sfreq, montage) if config.use_ica else None
        self.session_norm = SessionNormalizer(self.n_channels) if config.session_normalize else None

        # Calibration state
        self._calibration_buffer = np.zeros((self.n_channels, 0))
        self._is_calibrated = False

        self._design_filters()

        logger.info(
            "EEGProcessor initialized: %s (%d ch, %d Hz, window=%ds, ICA=%s)",
            config.headset, self.n_channels, self.sfreq,
            config.window_seconds, config.use_ica,
        )

    def _design_filters(self):
        """Pre-compute filter coefficients."""
        self._hann = np.hanning(self.window_samples)
        self._freqs = np.fft.rfftfreq(self.window_samples, 1.0 / self.sfreq)

        self._band_masks = {}
        for band in EEGBand:
            low, high = band.value
            mask = (self._freqs >= low) & (self._freqs <= high)
            self._band_masks[band.name] = mask

    def calibrate(self, data: np.ndarray):
        """Run full calibration from resting-state EEG.

        This MUST be called before real-time decoding. It:
        1. Fits ICA to identify artifact components
        2. Computes baseline statistics for artifact detection
        3. Computes session normalization parameters
        """
        logger.info("Calibrating EEG processor with %d samples (%.1fs)...",
                     data.shape[1], data.shape[1] / self.sfreq)

        # Pre-filter
        filtered = self._bandpass_filter(data)
        filtered = self._notch_filter(filtered)

        # Fit ICA
        if self.ica is not None:
            self.ica.fit(filtered)

        # Calibrate artifact detector
        if self.ica is not None and self.ica.is_fitted:
            clean = self.ica.remove_artifacts(filtered)
        else:
            clean = filtered
        self.artifact_detector.calibrate(clean)

        # Calibrate session normalizer
        if self.session_norm is not None:
            car_data = clean - clean.mean(axis=0, keepdims=True)
            band_powers = self._compute_band_powers(car_data)
            self.session_norm.calibrate(car_data, band_powers)

        self._is_calibrated = True
        logger.info("Calibration complete")

    def push_calibration_chunk(self, data: np.ndarray):
        """Push EEG data into the calibration buffer."""
        if data.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {data.shape[0]}")
        self._calibration_buffer = np.concatenate([self._calibration_buffer, data], axis=1)
        min_samples = int(self.config.ica_calibration_seconds * self.sfreq)
        if self._calibration_buffer.shape[1] >= min_samples and not self._is_calibrated:
            self.calibrate(self._calibration_buffer)

    def push_chunk(self, data: np.ndarray):
        """Push a chunk of raw EEG data into the processing buffer."""
        if data.shape[0] != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} channels, got {data.shape[0]}")
        self._buffer = np.concatenate([self._buffer, data], axis=1)

        if not self._is_calibrated:
            self.push_calibration_chunk(data)

    def has_window(self) -> bool:
        return self._buffer.shape[1] >= self.window_samples

    def process_window(self) -> Optional[dict]:
        """Process one window through the full production pipeline."""
        if not self.has_window():
            return None

        window = self._buffer[:, :self.window_samples].copy()
        step = self.window_samples - self.overlap_samples
        self._buffer = self._buffer[:, step:]
        self._total_windows += 1

        # --- Step 1: Bandpass + Notch ---
        window = self._bandpass_filter(window)
        window = self._notch_filter(window)

        # --- Step 2: ICA artifact removal ---
        if self.ica is not None and self.ica.is_fitted:
            window = self.ica.remove_artifacts(window)

        # --- Step 3: Detect remaining artifacts ---
        artifacts = self.artifact_detector.detect(window)
        artifact_summary = self.artifact_detector.get_artifact_summary(
            artifacts, self.config.window_seconds
        )

        # Strict mode: reject entire window if contaminated
        if self.config.strict_artifact_rejection and not artifact_summary["is_clean"]:
            self._rejected_windows += 1
            return {
                "rejected": True,
                "reason": f"artifact: {artifact_summary['dominant_artifact']}",
                "artifact_summary": artifact_summary,
                "quality": {"overall": 0.0, "clean_ratio": 0.0,
                           "channel_snr_db": [0.0] * self.n_channels,
                           "mean_snr_db": 0.0, "artifact_channels": self.channels},
            }

        # Interpolate artifact-contaminated samples
        window, amplitude_mask = self._amplitude_rejection(window)
        combined_mask = artifacts["any"] | amplitude_mask

        # --- Step 4: CAR ---
        if self.config.apply_car:
            window = self._common_average_reference(window)

        # --- Step 5: Session normalization ---
        if self.session_norm is not None and self.session_norm.is_calibrated:
            window = self.session_norm.normalize(window)

        # --- Step 6: Feature extraction ---
        features = {
            "raw_filtered": window,
            "artifact_mask": combined_mask,
            "artifact_summary": artifact_summary,
            "rejected": False,
        }

        if self.config.compute_band_powers:
            bp = self._compute_band_powers(window)
            if self.session_norm is not None and self.session_norm.is_calibrated:
                features["band_powers_raw"] = bp
                features["band_powers"] = self.session_norm.normalize_band_powers(bp)
            else:
                features["band_powers"] = bp

        if self.config.compute_phase:
            features["phase"] = self._compute_instantaneous_phase(window)

        if self.config.compute_connectivity and self.n_channels >= 2:
            features["connectivity"] = self._compute_phase_connectivity(window)

        features["quality"] = self._estimate_signal_quality(window, combined_mask, artifact_summary)

        return features

    def process_all(self) -> list[dict]:
        results = []
        while self.has_window():
            result = self.process_window()
            if result is not None:
                results.append(result)
        return results

    def _bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, 1.0 / self.sfreq)
        filt = np.ones(len(freqs))

        low = self.config.bandpass_low
        transition = low * 0.5
        mask_low = freqs < low
        mask_transition = (freqs >= low - transition) & (freqs < low)
        filt[mask_low] = 0.0
        if np.any(mask_transition):
            filt[mask_transition] = 0.5 * (
                1 + np.cos(np.pi * (freqs[mask_transition] - low) / transition)
            )

        high = self.config.bandpass_high
        transition = high * 0.1
        mask_high = freqs > high + transition
        mask_transition = (freqs >= high) & (freqs <= high + transition)
        filt[mask_high] = 0.0
        if np.any(mask_transition):
            filt[mask_transition] = 0.5 * (
                1 + np.cos(np.pi * (freqs[mask_transition] - high) / transition)
            )

        result = np.zeros_like(data)
        for ch in range(data.shape[0]):
            spectrum = np.fft.rfft(data[ch])
            spectrum *= filt
            result[ch] = np.fft.irfft(spectrum, n=n)
        return result

    def _notch_filter(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, 1.0 / self.sfreq)
        notch = np.ones(len(freqs))
        fundamental = self.config.notch_freq
        width = self.config.notch_width

        for harmonic in range(1, 4):
            center = fundamental * harmonic
            if center >= self.sfreq / 2:
                break
            mask = np.abs(freqs - center) < width
            notch[mask] = 0.0
            transition_mask = (np.abs(freqs - center) >= width) & (
                np.abs(freqs - center) < width * 2
            )
            if np.any(transition_mask):
                dist = np.abs(freqs[transition_mask] - center)
                notch[transition_mask] = (dist - width) / width

        result = np.zeros_like(data)
        for ch in range(data.shape[0]):
            spectrum = np.fft.rfft(data[ch])
            spectrum *= notch
            result[ch] = np.fft.irfft(spectrum, n=n)
        return result

    def _amplitude_rejection(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Reject remaining amplitude outliers that ICA missed."""
        threshold = self.config.artifact_threshold_uv
        artifact_mask = np.abs(data) > threshold

        gradient = np.diff(data, axis=1, prepend=data[:, :1])
        artifact_mask |= np.abs(gradient) > threshold * 0.5

        cleaned = data.copy()
        for ch in range(data.shape[0]):
            bad = np.where(artifact_mask[ch])[0]
            if len(bad) == 0:
                continue
            good = np.where(~artifact_mask[ch])[0]
            if len(good) < 2:
                cleaned[ch] = 0.0
                continue
            cleaned[ch, bad] = np.interp(bad, good, data[ch, good])

        return cleaned, artifact_mask.any(axis=0)

    def _common_average_reference(self, data: np.ndarray) -> np.ndarray:
        return data - data.mean(axis=0, keepdims=True)

    def _compute_band_powers(self, data: np.ndarray) -> dict[str, np.ndarray]:
        band_powers = {}
        n = data.shape[1]
        hann = np.hanning(n)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sfreq)

        band_masks = {}
        for band in EEGBand:
            low, high = band.value
            band_masks[band.name] = (freqs >= low) & (freqs <= high)

        for ch in range(data.shape[0]):
            windowed = data[ch] * hann
            spectrum = np.abs(np.fft.rfft(windowed)) ** 2 / n
            for band_name, mask in band_masks.items():
                if band_name not in band_powers:
                    band_powers[band_name] = np.zeros(data.shape[0])
                band_powers[band_name][ch] = spectrum[mask].mean() if mask.any() else 0.0

        for band_name in band_powers:
            band_powers[band_name] = np.log1p(band_powers[band_name])
        return band_powers

    def _compute_instantaneous_phase(self, data: np.ndarray) -> np.ndarray:
        n = data.shape[1]
        phases = np.zeros_like(data)
        for ch in range(data.shape[0]):
            spectrum = np.fft.fft(data[ch])
            n_fft = len(spectrum)
            h = np.zeros(n_fft)
            h[0] = 1
            h[1:(n_fft + 1) // 2] = 2
            if n_fft % 2 == 0:
                h[n_fft // 2] = 1
            analytic = np.fft.ifft(spectrum * h)
            phases[ch] = np.angle(analytic)
        return phases

    def _compute_phase_connectivity(self, data: np.ndarray) -> np.ndarray:
        phases = self._compute_instantaneous_phase(data)
        n = self.n_channels
        connectivity = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                phase_diff = phases[i] - phases[j]
                plv = np.abs(np.mean(np.exp(1j * phase_diff)))
                connectivity[i, j] = plv
                connectivity[j, i] = plv
        idx = np.triu_indices(n, k=1)
        return connectivity[idx]

    def _estimate_signal_quality(
        self, data: np.ndarray, artifact_mask: np.ndarray,
        artifact_summary: Optional[dict] = None,
    ) -> dict:
        clean_ratio = 1.0 - artifact_mask.mean()

        channel_snr = np.zeros(self.n_channels)
        for ch in range(self.n_channels):
            signal_power = np.var(data[ch])
            spectrum = np.abs(np.fft.rfft(data[ch])) ** 2
            freqs = np.fft.rfftfreq(data.shape[1], 1.0 / self.sfreq)
            noise_power = spectrum[freqs > 40].mean() if (freqs > 40).any() else 1e-10
            channel_snr[ch] = 10 * np.log10(signal_power / max(noise_power, 1e-10))

        snr_score = np.clip(channel_snr.mean() / 20.0, 0, 1)

        # Penalize quality score for artifact contamination
        artifact_penalty = 0.0
        if artifact_summary:
            artifact_penalty = artifact_summary["total_contaminated_ratio"] * 0.3

        quality_score = max(0.0, clean_ratio * 0.4 + snr_score * 0.4 - artifact_penalty + 0.2)

        result = {
            "overall": float(quality_score),
            "clean_ratio": float(clean_ratio),
            "channel_snr_db": channel_snr.tolist(),
            "mean_snr_db": float(channel_snr.mean()),
            "artifact_channels": [
                self.channels[i]
                for i in range(self.n_channels)
                if self._artifact_counts[i] > self._total_windows * 0.3
            ],
            "ica_fitted": self.ica.is_fitted if self.ica else False,
            "session_calibrated": self._is_calibrated,
            "windows_rejected_ratio": self._rejected_windows / max(self._total_windows, 1),
        }

        if artifact_summary:
            result["artifact_detail"] = artifact_summary

        return result

    def get_feature_vector(self, features: dict) -> np.ndarray:
        parts = []
        parts.append(features["raw_filtered"].flatten())
        if "band_powers" in features:
            for band_name in sorted(features["band_powers"].keys()):
                parts.append(features["band_powers"][band_name])
        if "phase" in features:
            subsample = 8
            parts.append(features["phase"][:, ::subsample].flatten())
        if "connectivity" in features:
            parts.append(features["connectivity"])
        return np.concatenate(parts).astype(np.float32)

    @property
    def feature_dim(self) -> int:
        n = self.n_channels
        w = self.window_samples
        dim = n * w
        dim += n * len(EEGBand)
        dim += n * (w // 8)
        dim += n * (n - 1) // 2
        return dim

    def reset(self):
        self._buffer = np.zeros((self.n_channels, 0))
        self._calibration_buffer = np.zeros((self.n_channels, 0))
        self._artifact_counts = np.zeros(self.n_channels, dtype=int)
        self._total_windows = 0
        self._rejected_windows = 0
        self._is_calibrated = False

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def rejection_rate(self) -> float:
        if self._total_windows == 0:
            return 0.0
        return self._rejected_windows / self._total_windows
