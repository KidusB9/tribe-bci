"""
Emotiv headset interface.

Supports Emotiv EPOC, EPOC X, EPOC+, and Insight headsets.
Uses the Cortex API (via cortex Python library) or LSL streams.

The EPOC has 14 channels spread across the scalp, giving much better
spatial coverage than the Muse. This means better source localization
and more information for the domain adapter.
"""

import numpy as np
import threading
import time
import logging
from typing import Optional

from reverse_bci.headsets.base import BaseHeadset, HeadsetInfo

logger = logging.getLogger(__name__)

EMOTIV_MODELS = {
    "epoc": {
        "channels": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        ],
        "sfreq": 128,
    },
    "epoc_x": {
        "channels": [
            "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
            "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
        ],
        "sfreq": 256,
    },
    "insight": {
        "channels": ["AF3", "AF4", "T7", "T8", "Pz"],
        "sfreq": 128,
    },
}


class EmotivHeadset(BaseHeadset):
    """Interface for Emotiv EEG headsets.

    Connection methods:
    1. Cortex API (requires Emotiv Launcher and account)
    2. LSL stream (via EmotivPRO or third-party tools)
    """

    def __init__(self, model: str = "epoc", client_id: Optional[str] = None, client_secret: Optional[str] = None):
        super().__init__()
        if model not in EMOTIV_MODELS:
            raise ValueError(f"Unknown Emotiv model: {model}. Supported: {list(EMOTIV_MODELS.keys())}")
        self.model = model
        self.model_config = EMOTIV_MODELS[model]
        self.client_id = client_id
        self.client_secret = client_secret
        self._inlet = None
        self._cortex = None

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to Emotiv headset."""
        # Try LSL first
        if self._connect_lsl(timeout):
            self._is_connected = True
            logger.info("Connected to Emotiv via LSL")
            return True

        # Try Cortex API
        if self.client_id and self._connect_cortex(timeout):
            self._is_connected = True
            logger.info("Connected to Emotiv via Cortex API")
            return True

        logger.error(
            "Could not connect to Emotiv. Options:\n"
            "  1. Start EmotivPRO with LSL output enabled\n"
            "  2. Provide client_id/client_secret for Cortex API\n"
            "  Install: pip install cortex"
        )
        return False

    def _connect_lsl(self, timeout: float) -> bool:
        """Connect via Lab Streaming Layer."""
        try:
            import pylsl

            streams = pylsl.resolve_byprop("type", "EEG", timeout=min(timeout, 5.0))
            n_ch = len(self.model_config["channels"])

            for stream in streams:
                name_lower = stream.name().lower()
                if "emotiv" in name_lower or stream.channel_count() == n_ch:
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

    def _connect_cortex(self, timeout: float) -> bool:
        """Connect via Emotiv Cortex API."""
        try:
            from cortex import Cortex

            self._cortex = Cortex(
                client_id=self.client_id,
                client_secret=self.client_secret,
            )
            self._cortex.open()
            # Query headsets
            headsets = self._cortex.query_headsets()
            if not headsets:
                return False
            self._cortex.connect_headset(headsets[0]["id"])
            self._cortex.create_session()
            self._cortex.subscribe(["eeg"])
            return True
        except ImportError:
            return False
        except Exception as e:
            logger.debug("Cortex connection failed: %s", e)
            return False

    def disconnect(self):
        self.stop_streaming()
        if self._inlet:
            self._inlet.close_stream()
            self._inlet = None
        if self._cortex:
            try:
                self._cortex.close()
            except Exception:
                pass
            self._cortex = None
        self._is_connected = False

    def get_info(self) -> HeadsetInfo:
        cfg = self.model_config
        return HeadsetInfo(
            name=f"Emotiv {self.model.upper()}",
            channels=cfg["channels"],
            sfreq=cfg["sfreq"],
            n_channels=len(cfg["channels"]),
        )

    def _read_chunk(self) -> Optional[np.ndarray]:
        if self._inlet:
            try:
                samples, _ = self._inlet.pull_chunk(timeout=0.1, max_samples=32)
                if not samples:
                    return None
                return np.array(samples).T
            except Exception:
                return None

        if self._cortex:
            try:
                data = self._cortex.get_data(timeout=0.1)
                if data and "eeg" in data:
                    return np.array(data["eeg"]).T
            except Exception:
                return None

        return None
