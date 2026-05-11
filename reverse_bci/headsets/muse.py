"""
Muse headset interface.

Supports Muse 2 and Muse S via Bluetooth Low Energy (BLE).
The Muse has 4 EEG channels: TP9, AF7, AF8, TP10.

TP9/TP10 are behind the ears (temporal) - good for auditory/speech processing
AF7/AF8 are on the forehead (frontal) - good for attention, language semantics

Uses the `muselsl` library for BLE communication, or falls back to
pylsl (Lab Streaming Layer) if muselsl streams are already active.
"""

import numpy as np
import threading
import time
import logging
from typing import Optional

from reverse_bci.headsets.base import BaseHeadset, HeadsetInfo

logger = logging.getLogger(__name__)

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
MUSE_SFREQ = 256


class MuseHeadset(BaseHeadset):
    """Interface for Muse 2 / Muse S EEG headsets.

    Connection methods (tried in order):
    1. Direct BLE via muselsl
    2. LSL stream (if muselsl is already streaming)
    3. Bluetooth serial (fallback)
    """

    def __init__(self, device_name: Optional[str] = None):
        super().__init__()
        self.device_name = device_name
        self._inlet = None
        self._buffer = np.zeros((4, 0))
        self._lock = threading.Lock()

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to Muse headset."""
        # Try LSL first (if muselsl is already streaming)
        if self._connect_lsl(timeout):
            self._is_connected = True
            logger.info("Connected to Muse via LSL")
            return True

        # Try direct BLE
        if self._connect_ble(timeout):
            self._is_connected = True
            logger.info("Connected to Muse via BLE")
            return True

        logger.error(
            "Could not connect to Muse. Ensure the headset is on and "
            "muselsl is installed: pip install muselsl"
        )
        return False

    def _connect_lsl(self, timeout: float) -> bool:
        """Connect via Lab Streaming Layer."""
        try:
            import pylsl

            streams = pylsl.resolve_byprop(
                "type", "EEG", timeout=min(timeout, 5.0)
            )
            if not streams:
                return False

            # Find Muse stream
            for stream in streams:
                if "muse" in stream.name().lower() or stream.channel_count() == 4:
                    self._inlet = pylsl.StreamInlet(
                        stream, max_buflen=360, max_chunklen=12
                    )
                    return True
            return False
        except ImportError:
            return False
        except Exception as e:
            logger.debug("LSL connection failed: %s", e)
            return False

    def _connect_ble(self, timeout: float) -> bool:
        """Connect via Bluetooth Low Energy using muselsl."""
        try:
            from muselsl import stream, list_muses

            muses = list_muses()
            if not muses:
                logger.info("No Muse devices found via BLE scan")
                return False

            # Pick the specified device or the first one found
            target = None
            for muse in muses:
                if self.device_name is None or muse["name"] == self.device_name:
                    target = muse
                    break

            if target is None:
                return False

            # Start muselsl stream in background
            self._stream_thread_ble = threading.Thread(
                target=stream,
                args=(target["address"],),
                daemon=True,
            )
            self._stream_thread_ble.start()
            time.sleep(2)  # Wait for stream to start

            # Now connect via LSL
            return self._connect_lsl(timeout)

        except ImportError:
            logger.debug("muselsl not installed")
            return False
        except Exception as e:
            logger.debug("BLE connection failed: %s", e)
            return False

    def disconnect(self):
        """Disconnect from Muse."""
        self.stop_streaming()
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None
        self._is_connected = False

    def get_info(self) -> HeadsetInfo:
        return HeadsetInfo(
            name="Muse",
            channels=MUSE_CHANNELS,
            sfreq=MUSE_SFREQ,
            n_channels=4,
        )

    def _read_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of EEG from the Muse."""
        if self._inlet is None:
            return None

        try:
            samples, timestamps = self._inlet.pull_chunk(
                timeout=0.1, max_samples=32
            )
            if not samples:
                return None

            # samples is list of [ch1, ch2, ch3, ch4] lists
            data = np.array(samples).T  # (4, n_samples)
            return data

        except Exception as e:
            logger.debug("Read error: %s", e)
            return None
