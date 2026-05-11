"""User interface components for the Reverse BCI application."""

from reverse_bci.ui.app import BCIApp

__all__ = ["BCIApp", "PredictiveSpellerWindow"]


def __getattr__(name):
    if name == "PredictiveSpellerWindow":
        from reverse_bci.ui.predictive_speller import PredictiveSpellerWindow
        return PredictiveSpellerWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
