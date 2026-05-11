"""
Simulated EEG headset for testing and development.

Generates synthetic EEG data that mimics consumer headset characteristics:
- Realistic noise floor (~1-5 uV RMS)
- Alpha rhythm (8-13 Hz, especially from posterior channels)
- Eye blink artifacts (every 3-7 seconds)
- Muscle artifacts (random bursts)
- Power line interference (50/60 Hz)

Can also embed known "neural signals" that the decoder should recover,
enabling end-to-end testing of the pipeline without a real headset.
"""

import numpy as np
import time
import logging
from typing import Optional

from reverse_bci.headsets.base import BaseHeadset, HeadsetInfo

logger = logging.getLogger(__name__)


class SimulatedHeadset(BaseHeadset):
    """Simulated EEG headset for testing the BCI pipeline.

    Generates realistic-looking EEG with known embedded signals,
    allowing end-to-end testing without hardware.
    """

    def __init__(
        self,
        headset_type: str = "muse",
        noise_level: float = 1.0,
        alpha_power: float = 5.0,
        blink_interval: float = 5.0,
        embed_signal: bool = True,
        target_word_index: int = 0,
        sfreq_override: Optional[int] = None,
    ):
        super().__init__()
        self.headset_type = headset_type
        self.noise_level = noise_level
        self.alpha_power = alpha_power
        self.blink_interval = blink_interval
        self.embed_signal = embed_signal
        self.target_word_index = target_word_index

        from reverse_bci.eeg_processor import HEADSET_MONTAGES
        montage = HEADSET_MONTAGES.get(headset_type)
        if montage is None:
            raise ValueError(f"Unknown headset type: {headset_type}")

        self.channels = montage["channels"]
        self.sfreq = sfreq_override or montage["sfreq"]
        self.n_channels = montage["n_channels"]

        self._time = 0.0
        self._last_read = time.time()
        self._rng = np.random.RandomState(42)
        self._last_blink = 0.0

    def connect(self, timeout: float = 10.0) -> bool:
        self._is_connected = True
        self._time = 0.0
        self._last_read = time.time()
        logger.info("Connected to simulated %s headset", self.headset_type)
        return True

    def disconnect(self):
        self.stop_streaming()
        self._is_connected = False

    def get_info(self) -> HeadsetInfo:
        return HeadsetInfo(
            name=f"Simulated {self.headset_type}",
            channels=self.channels,
            sfreq=self.sfreq,
            n_channels=self.n_channels,
            battery_level=1.0,
        )

    def _read_chunk(self) -> Optional[np.ndarray]:
        now = time.time()
        elapsed = now - self._last_read
        self._last_read = now

        # Generate samples for elapsed time
        n_samples = max(1, int(elapsed * self.sfreq))
        n_samples = min(n_samples, self.sfreq)  # Cap at 1 second

        data = self._generate_eeg(n_samples)
        self._time += n_samples / self.sfreq

        return data

    def _generate_eeg(self, n_samples: int) -> np.ndarray:
        """Generate realistic synthetic EEG data."""
        t = np.arange(n_samples) / self.sfreq + self._time
        data = np.zeros((self.n_channels, n_samples))

        # 1. Background noise (pink noise / 1/f)
        for ch in range(self.n_channels):
            white = self._rng.randn(n_samples) * self.noise_level
            # Simple 1/f approximation
            fft = np.fft.rfft(white)
            freqs = np.fft.rfftfreq(n_samples, 1.0 / self.sfreq)
            freqs[0] = 1.0  # Avoid division by zero
            fft /= np.sqrt(freqs)
            data[ch] = np.fft.irfft(fft, n=n_samples)

        # 2. Alpha rhythm (8-13 Hz) - strongest in posterior channels
        alpha_freq = 10.0 + self._rng.randn() * 0.5
        alpha = np.sin(2 * np.pi * alpha_freq * t) * self.alpha_power
        for ch in range(self.n_channels):
            # Posterior channels get more alpha
            ch_name = self.channels[ch] if ch < len(self.channels) else ""
            if any(p in ch_name for p in ["O", "P", "T"]):
                data[ch] += alpha * (0.8 + 0.4 * self._rng.rand())
            else:
                data[ch] += alpha * (0.2 + 0.2 * self._rng.rand())

        # 3. Eye blink artifacts (big, slow deflection on frontal channels)
        blink_elapsed = self._time - self._last_blink
        if blink_elapsed > self.blink_interval:
            # Generate blink
            blink_duration = int(0.3 * self.sfreq)
            if n_samples > blink_duration:
                blink_start = self._rng.randint(0, n_samples - blink_duration)
                blink_t = np.arange(blink_duration) / self.sfreq
                blink_shape = 50 * np.exp(-((blink_t - 0.15) ** 2) / 0.005)

                for ch in range(self.n_channels):
                    ch_name = self.channels[ch] if ch < len(self.channels) else ""
                    if any(p in ch_name for p in ["AF", "Fp", "F"]):
                        data[ch, blink_start:blink_start + blink_duration] += blink_shape
                    else:
                        data[ch, blink_start:blink_start + blink_duration] += blink_shape * 0.2

                self._last_blink = self._time

        # 4. Power line interference (60 Hz)
        line_noise = 0.5 * np.sin(2 * np.pi * 60.0 * t)
        data += line_noise[np.newaxis, :]

        # 5. Embedded neural signal (for testing decoder recovery)
        if self.embed_signal:
            data = self._embed_target_signal(data, t)

        return data

    def _embed_target_signal(self, data: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Embed a weak but detectable signal pattern corresponding to a target word.

        The embedded signal is subtle (realistic for actual neural activity)
        but has a consistent pattern that the trained decoder should recover.
        """
        n_samples = data.shape[1]

        # Create a word-specific spatial-temporal pattern
        np.random.seed(self.target_word_index * 137 + 42)
        spatial_pattern = np.random.randn(self.n_channels)
        spatial_pattern /= np.linalg.norm(spatial_pattern)

        # Temporal pattern: word-specific frequency modulation
        base_freq = 4.0 + (self.target_word_index % 8)  # 4-12 Hz
        temporal = np.sin(2 * np.pi * base_freq * t)

        # Amplitude: very weak (realistic neural signal level)
        amplitude = 0.5  # microvolts

        signal = amplitude * np.outer(spatial_pattern, temporal)
        data += signal[:, :n_samples]

        np.random.seed(None)  # Reset seed
        return data

    def set_target_word(self, word_index: int):
        """Change the target word being "thought" for simulation."""
        self.target_word_index = word_index
        logger.info("Simulated target word index: %d", word_index)
