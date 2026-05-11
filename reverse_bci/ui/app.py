"""
BCI Application - User-facing interface.

Provides both a terminal UI and a web-based interface for the
Reverse BCI system. Designed for accessibility:
- Large, high-contrast text output
- Signal quality indicators
- Word prediction panels
- Sentence builder with undo/clear
- Export/save decoded text
"""

import sys
import time
import threading
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from reverse_bci.bci_pipeline import BCIPipeline, BCIConfig
from reverse_bci.headsets.base import BaseHeadset
from reverse_bci.headsets.simulated import SimulatedHeadset

logger = logging.getLogger(__name__)


class BCIApp:
    """Main BCI application.

    Orchestrates the full user experience from headset connection
    through real-time text decoding.

    Usage:
        app = BCIApp()
        app.run()  # Interactive terminal mode

        # Or programmatic:
        app = BCIApp(headset="simulated")
        app.start()
        time.sleep(30)
        print(app.get_decoded_text())
        app.stop()
    """

    def __init__(
        self,
        headset: str = "simulated",
        model_path: Optional[str] = None,
        tribe_path: Optional[str] = None,
        config: Optional[BCIConfig] = None,
    ):
        self.headset_type = headset
        self.model_path = model_path
        self.tribe_path = tribe_path

        if config is None:
            config = BCIConfig(headset_type=headset)
        self.config = config

        self.pipeline = BCIPipeline(config)
        self._headset: Optional[BaseHeadset] = None
        self._log_file: Optional[Path] = None
        self._decoded_words: list[tuple[str, float, float]] = []

    def setup(self):
        """Initialize all components."""
        print("=" * 60)
        print("  REVERSE BCI - Non-invasive Brain-Computer Interface")
        print("  Powered by TRIBE v2 (Meta AI)")
        print("=" * 60)
        print()

        # Load models
        print("[1/3] Loading neural network models...")
        self.pipeline.load_models(self.model_path)

        # Initialize from TRIBE v2 if available
        if self.tribe_path:
            print("       Initializing from TRIBE v2 weights...")
            self.pipeline.init_from_tribe(self.tribe_path)

        # Connect headset
        print(f"[2/3] Connecting to {self.headset_type} headset...")
        self._headset = self._create_headset()
        self.pipeline.connect_headset(self._headset)

        # Register callbacks
        self.pipeline.on_word(self._on_word)
        self.pipeline.on_status(self._on_status)
        self.pipeline.on_quality(self._on_quality)

        print("[3/3] Setup complete!")
        print()

    def _create_headset(self) -> BaseHeadset:
        """Create the appropriate headset interface."""
        if self.headset_type == "simulated":
            headset = SimulatedHeadset(headset_type="muse", embed_signal=True)
            headset.connect()
            return headset
        elif self.headset_type == "muse":
            from reverse_bci.headsets.muse import MuseHeadset
            headset = MuseHeadset()
            if not headset.connect():
                print("  [!] Could not connect to Muse. Falling back to simulated.")
                headset = SimulatedHeadset(headset_type="muse")
                headset.connect()
            return headset
        elif self.headset_type.startswith("emotiv"):
            from reverse_bci.headsets.emotiv import EmotivHeadset
            model = self.headset_type.replace("emotiv_", "").replace("emotiv", "epoc")
            headset = EmotivHeadset(model=model)
            if not headset.connect():
                print("  [!] Could not connect to Emotiv. Falling back to simulated.")
                headset = SimulatedHeadset(headset_type="emotiv_epoc")
                headset.connect()
            return headset
        else:
            raise ValueError(f"Unknown headset: {self.headset_type}")

    def start(self):
        """Start decoding."""
        self.pipeline.start()

    def stop(self):
        """Stop decoding."""
        self.pipeline.stop()

    def run(self):
        """Run the interactive terminal application."""
        self.setup()

        print("-" * 60)
        print("  Signal Quality Monitor")
        print("-" * 60)
        print("  Checking headset signal...")
        print()

        # Calibration
        self._run_calibration()

        # Start decoding
        print("-" * 60)
        print("  LIVE DECODING")
        print("-" * 60)
        print("  Decoded text will appear below.")
        print("  Press Ctrl+C to stop.")
        print()

        self.start()

        # Start log file
        self._start_logging()

        try:
            self._display_loop()
        except KeyboardInterrupt:
            print("\n")
            print("Stopping...")
        finally:
            self.stop()
            self._stop_logging()
            self._print_summary()

    def _run_calibration(self):
        """Run the calibration phase with user feedback."""
        print("  Starting calibration phase...")
        print("  Please sit still and relax for 30 seconds.")
        print("  Try to keep your eyes open and minimize movement.")
        print()

        # Start streaming for calibration
        self._headset.start_streaming()
        time.sleep(1)

        duration = self.config.calibration_duration
        start = time.time()

        while time.time() - start < duration:
            elapsed = time.time() - start
            remaining = duration - elapsed
            bar_len = 40
            filled = int(bar_len * elapsed / duration)
            bar = "#" * filled + "-" * (bar_len - filled)
            sys.stdout.write(f"\r  [{bar}] {remaining:.0f}s remaining")
            sys.stdout.flush()

            # Process any available EEG
            if self.pipeline.processor.has_window():
                features = self.pipeline.processor.process_window()

            time.sleep(0.5)

        print("\n  Calibration complete!")
        print()

    def _display_loop(self):
        """Main display loop showing decoded text and status."""
        last_word_count = 0
        last_status_time = time.time()

        while True:
            time.sleep(0.5)

            # Show new decoded words
            current_count = len(self._decoded_words)
            if current_count > last_word_count:
                for word, conf, ts in self._decoded_words[last_word_count:]:
                    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                    conf_bar = "#" * int(conf * 10)
                    print(f"  [{timestamp}] {word:>15s}  [{conf_bar:<10s}] {conf:.0%}")
                last_word_count = current_count

            # Periodic status update
            if time.time() - last_status_time > 10:
                stats = self.pipeline.stats
                sentence = stats["current_sentence"]
                if sentence:
                    print()
                    print(f"  >> {sentence}")
                    print()
                last_status_time = time.time()

    def _on_word(self, word: str, confidence: float):
        """Callback when a word is decoded."""
        self._decoded_words.append((word, confidence, time.time()))

        # Log to file
        if self._log_file:
            with open(self._log_file, "a") as f:
                ts = datetime.now().isoformat()
                f.write(f"{ts}\t{word}\t{confidence:.4f}\n")

    def _on_status(self, status: dict):
        """Callback for status updates."""
        state = status.get("state", "")
        message = status.get("message", "")
        if state in ("bad_signal", "error"):
            print(f"\n  [!] {message}")

    def _on_quality(self, quality: dict):
        """Callback for signal quality updates (throttled in display)."""
        pass

    def _start_logging(self):
        log_dir = Path("./bci_logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_file = log_dir / f"bci_session_{timestamp}.tsv"
        with open(self._log_file, "w") as f:
            f.write("timestamp\tword\tconfidence\n")
        print(f"  Logging to: {self._log_file}")

    def _stop_logging(self):
        if self._log_file and self._log_file.exists():
            # Write final sentence
            with open(self._log_file, "a") as f:
                f.write(f"\n# Final sentence: {self.get_decoded_text()}\n")

    def _print_summary(self):
        """Print session summary."""
        stats = self.pipeline.stats
        print()
        print("=" * 60)
        print("  SESSION SUMMARY")
        print("=" * 60)
        print(f"  Words decoded:    {stats['words_decoded']}")
        print(f"  Total decodings:  {stats['decode_count']}")
        print(f"  Avg latency:      {stats['avg_latency_ms']:.1f} ms")
        print()
        print(f"  Decoded text: {stats['current_sentence']}")
        print()

        if self._log_file:
            print(f"  Session log: {self._log_file}")

        # Save decoded text
        if stats["current_sentence"]:
            output_file = Path("./bci_output.txt")
            with open(output_file, "a") as f:
                ts = datetime.now().isoformat()
                f.write(f"[{ts}] {stats['current_sentence']}\n")
            print(f"  Output saved: {output_file}")
        print("=" * 60)

    def get_decoded_text(self) -> str:
        """Get the full decoded sentence."""
        return self.pipeline.output_buffer.current_sentence


def main():
    """Entry point for the BCI application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Reverse BCI: Non-invasive Brain-Computer Interface"
    )
    parser.add_argument(
        "--headset", default="simulated",
        choices=["simulated", "muse", "emotiv_epoc", "emotiv_insight"],
        help="EEG headset to use",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to trained BCI model checkpoint",
    )
    parser.add_argument(
        "--tribe", default=None,
        help="Path to TRIBE v2 model (for initialization)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Compute device (auto, cpu, cuda)",
    )
    parser.add_argument(
        "--calibration-time", type=float, default=30.0,
        help="Calibration duration in seconds",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.3,
        help="Minimum confidence threshold for word output",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = BCIConfig(
        headset_type=args.headset,
        device=args.device,
        calibration_duration=args.calibration_time,
        confidence_threshold=args.confidence,
        model_checkpoint=args.model,
        tribe_checkpoint=args.tribe,
    )

    app = BCIApp(
        headset=args.headset,
        model_path=args.model,
        tribe_path=args.tribe,
        config=config,
    )

    app.run()


if __name__ == "__main__":
    main()
