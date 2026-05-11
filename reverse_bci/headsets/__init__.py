"""
Consumer EEG headset interfaces.

Provides unified streaming API for multiple consumer EEG devices.
Each headset streams raw EEG data into the BCI pipeline.
"""

from reverse_bci.headsets.base import BaseHeadset
from reverse_bci.headsets.muse import MuseHeadset
from reverse_bci.headsets.emotiv import EmotivHeadset
from reverse_bci.headsets.simulated import SimulatedHeadset

__all__ = ["BaseHeadset", "MuseHeadset", "EmotivHeadset", "SimulatedHeadset"]
