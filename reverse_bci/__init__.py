"""
Reverse BCI: Non-invasive Brain-Computer Interface powered by TRIBE v2.

Decodes text from consumer EEG headsets by leveraging TRIBE v2's deep
understanding of how the brain processes language, vision, and audio.

Architecture:
    EEG (raw) -> ICA Artifact Rejection -> Preprocessing
    -> EEG Encoder (bottleneck proportional to channel count)
    -> Domain Adapter (few-shot calibrated per session)
    -> TRIBE Latent Space -> Reverse Decoder -> Text

Supports Muse, Emotiv, and other consumer EEG headsets.
"""

from reverse_bci.eeg_processor import EEGProcessor
from reverse_bci.eeg_encoder import EEGEncoder
from reverse_bci.domain_adapter import DomainAdapter
from reverse_bci.reverse_decoder import ReverseDecoder
from reverse_bci.text_decoder import TextDecoder
from reverse_bci.bci_pipeline import BCIPipeline
from reverse_bci.calibration import FewShotCalibrator

__all__ = [
    "EEGProcessor",
    "EEGEncoder",
    "DomainAdapter",
    "ReverseDecoder",
    "TextDecoder",
    "BCIPipeline",
    "FewShotCalibrator",
]
