"""
Real-time BCI Pipeline.

Orchestrates the full decode loop:
    Headset -> EEG Processor (ICA + artifact rejection)
    -> EEG Encoder (channel-aware bottleneck)
    -> Domain Adapter (few-shot calibrated per session)
    -> Reverse Decoder -> Text Decoder -> Output

Critical design choices:
- ICA artifact rejection runs BEFORE any neural network processing
- Few-shot calibration adapts to the user's brain every session
- Strict mode rejects artifact-contaminated windows entirely
- P300 speller mode available for paralyzed users
"""

import torch
import numpy as np
import threading
import time
import logging
from typing import Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque

from reverse_bci.eeg_processor import EEGProcessor, EEGProcessorConfig
from reverse_bci.eeg_encoder import EEGEncoder, EEGEncoderConfig
from reverse_bci.domain_adapter import DomainAdapter, DomainAdapterConfig
from reverse_bci.reverse_decoder import ReverseDecoder, ReverseDecoderConfig
from reverse_bci.text_decoder import TextDecoder, TextDecoderConfig
from reverse_bci.headsets.base import BaseHeadset

logger = logging.getLogger(__name__)


@dataclass
class BCIConfig:
    headset_type: str = "muse"
    device: str = "auto"
    window_seconds: float = 2.0
    confidence_threshold: float = 0.3
    output_mode: str = "word"  # "word", "letter", "phrase"
    use_language_model: bool = True
    calibration_duration: float = 30.0
    tribe_checkpoint: Optional[str] = None
    model_checkpoint: Optional[str] = None
    # Adaptive parameters
    adaptive_threshold: bool = True
    smoothing_window: int = 3
    # EEG processing
    eeg_config: EEGProcessorConfig = field(default_factory=EEGProcessorConfig)
    # Model configs
    encoder_config: EEGEncoderConfig = field(default_factory=EEGEncoderConfig)
    adapter_config: DomainAdapterConfig = field(default_factory=DomainAdapterConfig)
    decoder_config: ReverseDecoderConfig = field(default_factory=ReverseDecoderConfig)
    text_config: TextDecoderConfig = field(default_factory=TextDecoderConfig)


class OutputBuffer:
    """Buffers and smooths decoded outputs over time."""

    def __init__(self, smoothing_window: int = 3):
        self.smoothing_window = smoothing_window
        self.history: deque = deque(maxlen=100)
        self.word_history: list[str] = []
        self.logit_buffer: deque = deque(maxlen=smoothing_window)

    def add(self, decoded: dict):
        """Add a decoding result to the buffer."""
        self.history.append(decoded)
        if decoded.get("neural_logits") is not None:
            self.logit_buffer.append(decoded["neural_logits"])

    def get_smoothed_word(self) -> Optional[tuple[str, float]]:
        """Get the consensus word from recent decodings."""
        if not self.history:
            return None

        recent = list(self.history)[-self.smoothing_window:]
        words = [d["decoded_words"][0] if d.get("decoded_words") else None for d in recent]
        words = [w for w in words if w is not None]

        if not words:
            return None

        # Majority vote
        from collections import Counter
        counts = Counter(words)
        best_word, count = counts.most_common(1)[0]
        confidence = count / len(words)

        return best_word, confidence

    def get_smoothed_logits(self) -> Optional[torch.Tensor]:
        """Average logits over the smoothing window for better accuracy."""
        if not self.logit_buffer:
            return None
        return torch.stack(list(self.logit_buffer)).mean(dim=0)

    def commit_word(self, word: str):
        """Add a confirmed word to the output sequence."""
        self.word_history.append(word)

    @property
    def current_sentence(self) -> str:
        return " ".join(self.word_history)


class BCIPipeline:
    """Main BCI pipeline for real-time EEG-to-text decoding.

    Usage:
        pipeline = BCIPipeline(config)
        pipeline.load_models()
        pipeline.connect_headset(headset)
        pipeline.start()

        # Words appear via the on_word callback
        pipeline.on_word(lambda word, conf: print(f"{word} ({conf:.0%})"))
    """

    def __init__(self, config: Optional[BCIConfig] = None):
        if config is None:
            config = BCIConfig()
        self.config = config

        # Determine device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)

        # Initialize processing pipeline
        config.eeg_config.headset = config.headset_type
        self.processor = EEGProcessor(config.eeg_config)

        # Models (initialized in load_models)
        self.encoder: Optional[EEGEncoder] = None
        self.adapter: Optional[DomainAdapter] = None
        self.decoder: Optional[ReverseDecoder] = None
        self.text_decoder: Optional[TextDecoder] = None

        # State
        self.headset: Optional[BaseHeadset] = None
        self.output_buffer = OutputBuffer(config.smoothing_window)
        self._is_running = False
        self._stop_event = threading.Event()
        self._callbacks: list[Callable] = []
        self._status_callbacks: list[Callable] = []
        self._quality_callbacks: list[Callable] = []

        # Calibration
        self._is_calibrating = False
        self._calibration_data: list[np.ndarray] = []

        # Statistics
        self._decode_count = 0
        self._total_latency = 0.0

    def load_models(self, checkpoint_path: Optional[str] = None):
        """Load or initialize all neural network models."""
        path = checkpoint_path or self.config.model_checkpoint

        # Configure encoder for the headset
        from reverse_bci.eeg_processor import HEADSET_MONTAGES
        montage = HEADSET_MONTAGES[self.config.headset_type]
        self.config.encoder_config.n_channels = montage["n_channels"]
        self.config.encoder_config.n_samples = int(
            self.config.window_seconds * montage["sfreq"]
        )

        # Initialize models
        self.encoder = EEGEncoder(self.config.encoder_config).to(self.device)
        self.adapter = DomainAdapter(self.config.adapter_config).to(self.device)
        self.decoder = ReverseDecoder(self.config.decoder_config).to(self.device)
        self.text_decoder = TextDecoder(self.config.text_config).to(self.device)

        # Load pretrained weights if available
        if path and Path(path).exists():
            logger.info("Loading BCI model from %s", path)
            state = torch.load(path, map_location=self.device, weights_only=True)
            if "encoder" in state:
                self.encoder.load_state_dict(state["encoder"])
            if "adapter" in state:
                self.adapter.load_state_dict(state["adapter"])
            if "decoder" in state:
                self.decoder.load_state_dict(state["decoder"])
            if "text_decoder" in state:
                self.text_decoder.load_state_dict(state["text_decoder"])
            logger.info("Loaded pretrained BCI model")
        else:
            logger.info("No checkpoint found - using randomly initialized models")
            logger.info("Train the model first using reverse_bci.training")

        # Set to eval mode
        self.encoder.eval()
        self.adapter.eval()
        self.decoder.eval()
        self.text_decoder.eval()

        total_params = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.encoder, self.adapter, self.decoder, self.text_decoder]
        )
        logger.info(
            "BCI pipeline loaded: %d parameters on %s",
            total_params, self.device,
        )

    def init_from_tribe(self, tribe_model_or_path):
        """Initialize the reverse decoder from TRIBE v2's pretrained weights.

        This leverages TRIBE v2's learned brain-mapping knowledge to give
        the reverse decoder a much better starting point.
        """
        if isinstance(tribe_model_or_path, (str, Path)):
            from tribev2 import TribeModel
            logger.info("Loading TRIBE v2 from %s", tribe_model_or_path)
            tribe = TribeModel.from_pretrained(tribe_model_or_path)
            tribe_model = tribe._model
        else:
            tribe_model = tribe_model_or_path

        if self.decoder is not None:
            self.decoder.init_from_tribe(tribe_model)
            logger.info("Initialized reverse decoder from TRIBE v2 weights")

    def connect_headset(self, headset: BaseHeadset):
        """Connect a headset to the pipeline."""
        self.headset = headset
        if not headset.is_connected:
            if not headset.connect():
                raise RuntimeError("Failed to connect to headset")

        # Register data callback
        headset.on_data(self._on_eeg_data)

    def on_word(self, callback: Callable[[str, float], None]):
        """Register callback for decoded words. Called with (word, confidence)."""
        self._callbacks.append(callback)

    def on_status(self, callback: Callable[[dict], None]):
        """Register callback for status updates."""
        self._status_callbacks.append(callback)

    def on_quality(self, callback: Callable[[dict], None]):
        """Register callback for signal quality updates."""
        self._quality_callbacks.append(callback)

    def start(self):
        """Start the real-time decoding pipeline."""
        if self.encoder is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        if self.headset is None:
            raise RuntimeError("No headset connected. Call connect_headset() first.")

        self._stop_event.clear()
        self._is_running = True

        # Start headset streaming
        self.headset.start_streaming()

        # Start decode loop
        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True
        )
        self._decode_thread.start()

        self._emit_status({"state": "running", "message": "BCI pipeline started"})
        logger.info("BCI pipeline started")

    def stop(self):
        """Stop the pipeline."""
        self._stop_event.set()
        self._is_running = False
        if self.headset:
            self.headset.stop_streaming()
        if hasattr(self, "_decode_thread"):
            self._decode_thread.join(timeout=5.0)
        self._emit_status({"state": "stopped", "message": "BCI pipeline stopped"})

    def calibrate(self, duration: Optional[float] = None):
        """Run calibration phase to adapt to the user's brain signals.

        During calibration, collect baseline EEG to:
        1. Estimate noise floor per channel
        2. Detect alpha rhythm characteristics
        3. Set adaptive thresholds
        """
        duration = duration or self.config.calibration_duration
        logger.info("Starting calibration (%.0f seconds)...", duration)
        self._is_calibrating = True
        self._calibration_data = []

        self._emit_status({
            "state": "calibrating",
            "message": f"Calibrating... please relax for {duration:.0f}s",
            "duration": duration,
        })

        # Collect data for the calibration period
        start = time.time()
        while time.time() - start < duration:
            if self.processor.has_window():
                features = self.processor.process_window()
                if features:
                    self._calibration_data.append(features)
            time.sleep(0.1)

        # Analyze calibration data
        self._apply_calibration()
        self._is_calibrating = False

        self._emit_status({
            "state": "calibrated",
            "message": "Calibration complete",
            "n_windows": len(self._calibration_data),
        })

    def _apply_calibration(self):
        """Apply calibration results to the processing pipeline."""
        if not self._calibration_data:
            return

        # Compute baseline statistics
        all_qualities = [d["quality"] for d in self._calibration_data]
        mean_snr = np.mean([q["mean_snr_db"] for q in all_qualities])

        # Adjust artifact threshold based on observed amplitude distribution
        all_raw = np.concatenate(
            [d["raw_filtered"] for d in self._calibration_data], axis=1
        )
        amplitude_95th = np.percentile(np.abs(all_raw), 95)
        self.processor.config.artifact_threshold_uv = amplitude_95th * 2

        logger.info(
            "Calibration: mean SNR=%.1f dB, artifact threshold=%.1f uV",
            mean_snr, self.processor.config.artifact_threshold_uv,
        )

    def _on_eeg_data(self, data: np.ndarray):
        """Callback for incoming EEG data from the headset."""
        self.processor.push_chunk(data)

    def _decode_loop(self):
        """Main decode loop running in background thread."""
        while not self._stop_event.is_set():
            if not self.processor.has_window():
                time.sleep(0.05)
                continue

            if self._is_calibrating:
                time.sleep(0.1)
                continue

            start_time = time.time()

            try:
                result = self._decode_one_window()
                if result:
                    latency = time.time() - start_time
                    self._decode_count += 1
                    self._total_latency += latency
                    result["latency_ms"] = latency * 1000
                    self._handle_decode_result(result)
            except Exception as e:
                logger.error("Decode error: %s", e, exc_info=True)

    @torch.no_grad()
    def _decode_one_window(self) -> Optional[dict]:
        """Process one EEG window through the full pipeline."""
        # Step 1: Process EEG
        features = self.processor.process_window()
        if features is None:
            return None

        # Emit quality update
        self._emit_quality(features["quality"])

        # Check signal quality
        if features["quality"]["overall"] < 0.1:
            return {"status": "bad_signal", "quality": features["quality"]}

        # Step 2: Convert to tensor
        raw = torch.tensor(
            features["raw_filtered"], dtype=torch.float32
        ).unsqueeze(0).to(self.device)  # (1, C, T)

        # Step 3: EEG Encoder -> latent
        enc_output = self.encoder(raw)
        eeg_latent = enc_output["latent"]  # (1, 1152)

        # Step 4: Domain Adapter -> TRIBE-compatible latent
        tribe_latent = self.adapter(eeg_latent)  # (1, 1152)

        # Step 5: Reverse Decoder -> text features
        dec_output = self.decoder(tribe_latent)
        text_features = dec_output["text_features"]  # (1, text_feat_dim)

        # Step 6: Text Decoder -> words
        if self.config.use_language_model and self.output_buffer.word_history:
            decoded = self.text_decoder.decode_with_context(
                text_features, self.output_buffer.word_history
            )
        else:
            decoded = self.text_decoder(text_features)

        decoded["quality"] = features["quality"]
        decoded["status"] = "decoded"
        return decoded

    def _handle_decode_result(self, result: dict):
        """Handle a decode result: buffer, threshold, and emit."""
        if result.get("status") == "bad_signal":
            self._emit_status({
                "state": "bad_signal",
                "message": "Poor signal quality - check headset fit",
            })
            return

        self.output_buffer.add(result)

        # Get smoothed result
        smoothed = self.output_buffer.get_smoothed_word()
        if smoothed is None:
            return

        word, confidence = smoothed

        # Check confidence threshold
        threshold = self.config.confidence_threshold
        if self.config.adaptive_threshold and self._decode_count > 10:
            # Adapt threshold based on recent history
            recent_confs = [
                d.get("confidence", torch.tensor([0.0]))[0].item()
                if isinstance(d.get("confidence"), torch.Tensor)
                else 0.0
                for d in list(self.output_buffer.history)[-20:]
            ]
            if recent_confs:
                threshold = max(threshold, np.percentile(recent_confs, 60))

        if confidence >= threshold:
            self.output_buffer.commit_word(word)
            for cb in self._callbacks:
                try:
                    cb(word, confidence)
                except Exception as e:
                    logger.error("Word callback error: %s", e)

    def _emit_status(self, status: dict):
        for cb in self._status_callbacks:
            try:
                cb(status)
            except Exception as e:
                logger.error("Status callback error: %s", e)

    def _emit_quality(self, quality: dict):
        for cb in self._quality_callbacks:
            try:
                cb(quality)
            except Exception as e:
                logger.error("Quality callback error: %s", e)

    def save_models(self, path: str):
        """Save all model weights."""
        state = {}
        if self.encoder:
            state["encoder"] = self.encoder.state_dict()
        if self.adapter:
            state["adapter"] = self.adapter.state_dict()
        if self.decoder:
            state["decoder"] = self.decoder.state_dict()
        if self.text_decoder:
            state["text_decoder"] = self.text_decoder.state_dict()
        torch.save(state, path)
        logger.info("Saved BCI model to %s", path)

    @property
    def stats(self) -> dict:
        avg_latency = (
            self._total_latency / self._decode_count
            if self._decode_count > 0
            else 0
        )
        return {
            "decode_count": self._decode_count,
            "avg_latency_ms": avg_latency * 1000,
            "words_decoded": len(self.output_buffer.word_history),
            "current_sentence": self.output_buffer.current_sentence,
            "is_running": self._is_running,
        }
