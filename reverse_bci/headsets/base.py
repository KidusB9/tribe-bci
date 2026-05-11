"""
Base headset interface for consumer EEG devices.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import threading
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class HeadsetInfo:
    name: str
    channels: list[str]
    sfreq: int
    n_channels: int
    battery_level: float = 1.0
    signal_quality: dict[str, float] = None

    def __post_init__(self):
        if self.signal_quality is None:
            self.signal_quality = {ch: 0.0 for ch in self.channels}


class BaseHeadset(ABC):
    """Abstract base class for consumer EEG headset interfaces.

    Provides a unified streaming API. Subclasses implement device-specific
    connection and data acquisition.
    """

    def __init__(self):
        self._is_connected = False
        self._is_streaming = False
        self._callbacks: list[Callable] = []
        self._stream_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @abstractmethod
    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to the headset. Returns True on success."""
        ...

    @abstractmethod
    def disconnect(self):
        """Disconnect from the headset."""
        ...

    @abstractmethod
    def get_info(self) -> HeadsetInfo:
        """Get headset information."""
        ...

    @abstractmethod
    def _read_chunk(self) -> Optional[np.ndarray]:
        """Read a chunk of raw EEG data.
        Returns (n_channels, n_samples) or None if no data.
        """
        ...

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

    def on_data(self, callback: Callable[[np.ndarray], None]):
        """Register a callback for incoming EEG data chunks."""
        self._callbacks.append(callback)

    def start_streaming(self):
        """Start streaming EEG data in a background thread."""
        if not self._is_connected:
            raise RuntimeError("Must connect before streaming")
        if self._is_streaming:
            return

        self._stop_event.clear()
        self._is_streaming = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop, daemon=True
        )
        self._stream_thread.start()
        logger.info("Started streaming from %s", self.get_info().name)

    def stop_streaming(self):
        """Stop streaming."""
        if not self._is_streaming:
            return
        self._stop_event.set()
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
        self._is_streaming = False
        logger.info("Stopped streaming")

    def _stream_loop(self):
        """Background thread that reads data and dispatches to callbacks."""
        info = self.get_info()
        chunk_size = info.sfreq // 10  # ~100ms chunks

        while not self._stop_event.is_set():
            chunk = self._read_chunk()
            if chunk is not None and chunk.shape[1] > 0:
                for callback in self._callbacks:
                    try:
                        callback(chunk)
                    except Exception as e:
                        logger.error("Callback error: %s", e)
            else:
                time.sleep(0.01)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.stop_streaming()
        self.disconnect()
