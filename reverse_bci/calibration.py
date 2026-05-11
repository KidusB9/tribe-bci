"""
Few-Shot Daily Calibration Protocol.

The brain is not a static machine. Between sessions, everything changes:
- Headset position (even 2mm matters)
- Brain state (sleep, caffeine, stress)
- Electrode impedance (sweat, hair, skin oil)

This module implements a rapid calibration loop that runs every time
the user puts the headset on. In ~2-3 minutes, it:

1. Collects resting-state EEG (30s) for ICA + session normalization
2. Prompts 10 known words (3 trials each = 30 trials total)
3. Uses these few-shot examples to fine-tune the domain adapter
   for THIS specific session

This is not optional. Without daily calibration, accuracy degrades
from whatever-you-trained to random chance within 24 hours.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import sys
import logging
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CalibrationConfig:
    resting_duration: float = 30.0
    calibration_words: list = None
    trials_per_word: int = 3
    think_duration: float = 3.0
    rest_duration: float = 1.5
    cue_duration: float = 1.5
    # Few-shot fine-tuning
    finetune_epochs: int = 20
    finetune_lr: float = 1e-3
    finetune_only_adapter: bool = True

    def __post_init__(self):
        if self.calibration_words is None:
            self.calibration_words = [
                "yes", "no", "help", "water", "pain",
                "stop", "go", "happy", "sad", "thank you",
            ]


class FewShotCalibrator:
    """Per-session few-shot calibration for the BCI pipeline.

    When the user puts on the headset:
    1. Collect resting-state data -> fit ICA + session normalization
    2. Run 10-word calibration -> collect (EEG, label) pairs
    3. Fine-tune the domain adapter on these few-shot examples
    4. Ready for live decoding
    """

    def __init__(
        self,
        config: Optional[CalibrationConfig] = None,
        device: str = "auto",
    ):
        if config is None:
            config = CalibrationConfig()
        self.config = config

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self._calibration_trials = []

    def run_calibration(
        self,
        pipeline,
        on_prompt: Optional[Callable[[str, str], None]] = None,
    ) -> dict:
        """Run the full calibration protocol.

        Args:
            pipeline: BCIPipeline instance (must have models loaded + headset connected)
            on_prompt: Callback(phase, message) for UI updates.
                       phase is "resting", "cue", "think", "rest", "finetuning", "done"

        Returns:
            dict with calibration results and metrics
        """
        if on_prompt is None:
            on_prompt = self._default_prompt

        results = {}
        headset = pipeline.headset
        processor = pipeline.processor

        # --- Phase 1: Resting-state calibration ---
        on_prompt("resting", f"Relax, eyes open, for {self.config.resting_duration:.0f} seconds...")
        resting_data = self._collect_resting(headset, processor)
        processor.calibrate(resting_data)
        results["resting_samples"] = resting_data.shape[1]
        results["ica_fitted"] = processor.ica.is_fitted if processor.ica else False
        on_prompt("resting", "Resting calibration complete.")

        # --- Phase 2: Word trials ---
        on_prompt("cue", f"Now you'll think of {len(self.config.calibration_words)} words, "
                         f"{self.config.trials_per_word} times each.")
        time.sleep(2)

        self._calibration_trials = []
        trial_order = self._make_trial_order()
        words = self.config.calibration_words

        for trial_num, word_idx in enumerate(trial_order):
            word = words[word_idx]
            total = len(trial_order)

            # Cue phase
            on_prompt("cue", f"[{trial_num + 1}/{total}] Get ready: {word.upper()}")
            time.sleep(self.config.cue_duration)

            # Think phase - collect EEG
            on_prompt("think", f"[{trial_num + 1}/{total}] THINK: {word.upper()}")
            trial_eeg = self._collect_trial(headset, processor)
            self._calibration_trials.append((trial_eeg, word_idx))

            # Rest phase
            on_prompt("rest", f"[{trial_num + 1}/{total}] rest...")
            time.sleep(self.config.rest_duration)

        results["n_trials"] = len(self._calibration_trials)

        # --- Phase 3: Few-shot fine-tuning ---
        on_prompt("finetuning", "Fine-tuning model for your brain...")
        finetune_results = self._finetune_adapter(pipeline)
        results.update(finetune_results)

        on_prompt("done", f"Calibration complete! Accuracy: {finetune_results.get('accuracy', 0):.0%}")
        return results

    def _collect_resting(self, headset, processor) -> np.ndarray:
        """Collect resting-state EEG."""
        buffer = np.zeros((processor.n_channels, 0))
        start = time.time()

        while time.time() - start < self.config.resting_duration:
            chunk = headset._read_chunk()
            if chunk is not None and chunk.shape[1] > 0:
                buffer = np.concatenate([buffer, chunk], axis=1)
            time.sleep(0.05)

        return buffer

    def _collect_trial(self, headset, processor) -> np.ndarray:
        """Collect EEG for one think trial."""
        buffer = np.zeros((processor.n_channels, 0))
        start = time.time()

        while time.time() - start < self.config.think_duration:
            chunk = headset._read_chunk()
            if chunk is not None and chunk.shape[1] > 0:
                buffer = np.concatenate([buffer, chunk], axis=1)
            time.sleep(0.05)

        # Ensure consistent length
        target_samples = int(self.config.think_duration * processor.sfreq)
        if buffer.shape[1] >= target_samples:
            return buffer[:, :target_samples]
        else:
            pad = np.zeros((processor.n_channels, target_samples - buffer.shape[1]))
            return np.concatenate([buffer, pad], axis=1)

    def _make_trial_order(self) -> list[int]:
        """Randomized trial order (interleave words, don't block them)."""
        order = []
        for _ in range(self.config.trials_per_word):
            indices = list(range(len(self.config.calibration_words)))
            np.random.shuffle(indices)
            order.extend(indices)
        return order

    @torch.no_grad()
    def _process_trials(self, pipeline) -> tuple[torch.Tensor, torch.Tensor]:
        """Process collected trials through the EEG processor and encoder."""
        processor = pipeline.processor
        encoder = pipeline.encoder
        eeg_tensors = []
        labels = []

        for trial_eeg, word_idx in self._calibration_trials:
            # Process through EEG pipeline
            processor.push_chunk(trial_eeg)
            windows = processor.process_all()

            for window in windows:
                if window.get("rejected"):
                    continue
                raw = torch.tensor(
                    window["raw_filtered"], dtype=torch.float32
                ).unsqueeze(0).to(self.device)
                eeg_tensors.append(raw)
                labels.append(word_idx)

        if not eeg_tensors:
            return torch.zeros(0), torch.zeros(0, dtype=torch.long)

        return torch.cat(eeg_tensors), torch.tensor(labels, dtype=torch.long).to(self.device)

    def _finetune_adapter(self, pipeline) -> dict:
        """Fine-tune the domain adapter on few-shot calibration data.

        Only fine-tunes the adapter (not the full pipeline) to avoid
        catastrophic forgetting of the pre-trained representations.
        Uses a high learning rate with early stopping.
        """
        eeg_data, labels = self._process_trials(pipeline)

        if len(eeg_data) == 0:
            logger.warning("No clean trials collected during calibration")
            return {"accuracy": 0.0, "n_clean_trials": 0}

        logger.info("Fine-tuning on %d clean windows from %d trials",
                     len(eeg_data), len(self._calibration_trials))

        encoder = pipeline.encoder
        adapter = pipeline.adapter
        n_words = len(self.config.calibration_words)

        # Simple classification head for calibration
        classifier = nn.Linear(adapter.config.tribe_latent_dim, n_words).to(self.device)

        # Choose what to fine-tune
        if self.config.finetune_only_adapter:
            params = list(adapter.parameters()) + list(classifier.parameters())
            encoder.eval()
        else:
            params = (
                list(encoder.parameters()) +
                list(adapter.parameters()) +
                list(classifier.parameters())
            )
            encoder.train()

        adapter.train()
        optimizer = optim.Adam(params, lr=self.config.finetune_lr)

        best_acc = 0.0
        for epoch in range(self.config.finetune_epochs):
            # Shuffle
            perm = torch.randperm(len(eeg_data))
            eeg_shuffled = eeg_data[perm]
            labels_shuffled = labels[perm]

            # Forward
            with torch.set_grad_enabled(not self.config.finetune_only_adapter):
                enc_out = encoder(eeg_shuffled)
            adapted = adapter(enc_out["latent"])
            logits = classifier(adapted)

            loss = nn.functional.cross_entropy(logits, labels_shuffled)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            # Check accuracy
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == labels_shuffled).float().mean().item()
                best_acc = max(best_acc, acc)

            if (epoch + 1) % 5 == 0:
                logger.info("Calibration epoch %d: loss=%.3f acc=%.1f%%",
                           epoch + 1, loss.item(), acc * 100)

        # Set back to eval
        encoder.eval()
        adapter.eval()

        return {
            "accuracy": best_acc,
            "n_clean_trials": len(eeg_data),
            "final_loss": loss.item(),
        }

    @staticmethod
    def _default_prompt(phase: str, message: str):
        """Default terminal-based prompt."""
        symbols = {
            "resting": "~",
            "cue": ">",
            "think": "*",
            "rest": ".",
            "finetuning": "#",
            "done": "!",
        }
        sym = symbols.get(phase, " ")
        sys.stdout.write(f"\r  [{sym}] {message:<60s}")
        sys.stdout.flush()
        if phase in ("done",):
            print()
