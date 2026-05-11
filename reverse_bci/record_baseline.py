#!/usr/bin/env python3
"""
N=1 Real-World Baseline Recording Script.

Records EEG data while you think of specific words.
This gives you the REAL accuracy of the system on biological data
(expect ~15%, not the synthetic 47%).

Protocol:
    1. Put on headset
    2. 60s resting-state calibration (eyes open, relax)
    3. For each target word:
       - Screen shows the word for 2 seconds
       - You concentrate on thinking that word for 4 seconds
       - 2 second rest
       - Repeat N times per word
    4. Save all data as .npz

Usage:
    python -m reverse_bci.record_baseline --headset muse --words "yes,no,water,pain,help"
    python -m reverse_bci.record_baseline --headset simulated --trials 20
"""

import argparse
import logging
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("reverse_bci.record")


def run_recording(
    headset_type: str = "simulated",
    words: list[str] = None,
    trials_per_word: int = 100,
    think_duration: float = 4.0,
    rest_duration: float = 2.0,
    cue_duration: float = 2.0,
    calibration_duration: float = 60.0,
    output_dir: str = "./eeg_recordings",
):
    """Run the full recording protocol."""

    if words is None:
        words = ["yes", "no", "water", "pain", "help"]

    from reverse_bci.eeg_processor import EEGProcessorConfig, HEADSET_MONTAGES
    montage = HEADSET_MONTAGES[headset_type]
    sfreq = montage["sfreq"]
    n_channels = montage["n_channels"]

    # Connect headset
    print("\n" + "=" * 60)
    print("  EEG BASELINE RECORDING")
    print("=" * 60)
    print(f"  Headset: {headset_type} ({n_channels} ch, {sfreq} Hz)")
    print(f"  Words: {words}")
    print(f"  Trials per word: {trials_per_word}")
    print(f"  Think duration: {think_duration}s")
    print(f"  Total recording time: ~{len(words) * trials_per_word * (cue_duration + think_duration + rest_duration) / 60:.0f} min")
    print("=" * 60)

    headset = _create_headset(headset_type)
    if not headset.is_connected:
        if not headset.connect():
            print("ERROR: Could not connect to headset")
            return

    # Data collection buffers
    eeg_buffer = np.zeros((n_channels, 0))
    collection_active = False

    def on_data(chunk):
        nonlocal eeg_buffer, collection_active
        if collection_active:
            eeg_buffer = np.concatenate([eeg_buffer, chunk], axis=1)

    headset.on_data(on_data)
    headset.start_streaming()

    # --- Phase 1: Resting-state calibration ---
    print(f"\n[CALIBRATION] Sit still, eyes open, relax for {calibration_duration:.0f} seconds...")
    print("  This data is used to calibrate artifact rejection.")
    print("  Minimize eye blinks and jaw movement.\n")

    collection_active = True
    calibration_start = time.time()
    while time.time() - calibration_start < calibration_duration:
        elapsed = time.time() - calibration_start
        remaining = calibration_duration - elapsed
        bar_len = 40
        filled = int(bar_len * elapsed / calibration_duration)
        bar = "#" * filled + "-" * (bar_len - filled)
        sys.stdout.write(f"\r  [{bar}] {remaining:.0f}s ")
        sys.stdout.flush()
        time.sleep(0.5)
    print()

    calibration_data = eeg_buffer.copy()
    print(f"  Collected {calibration_data.shape[1]} calibration samples")

    # --- Phase 2: Word trials ---
    print(f"\n[RECORDING] Starting word trials...")
    print("  When you see a word, concentrate on THINKING it clearly.")
    print("  Try to 'hear' the word in your mind's voice.")
    print()

    all_trials = []  # List of (eeg_segment, word_index)
    trial_samples = int(think_duration * sfreq)

    total_trials = len(words) * trials_per_word
    trial_count = 0

    # Randomize trial order (don't present all of one word, then all of another)
    trial_order = []
    for _ in range(trials_per_word):
        indices = list(range(len(words)))
        np.random.shuffle(indices)
        trial_order.extend(indices)

    for trial_num, word_idx in enumerate(trial_order):
        word = words[word_idx]
        trial_count += 1

        # Cue
        sys.stdout.write(f"\r  Trial {trial_count}/{total_trials}:  >>> {word.upper():^15s} <<<  ")
        sys.stdout.flush()
        time.sleep(cue_duration)

        # Record thinking phase
        eeg_buffer = np.zeros((n_channels, 0))
        collection_active = True

        sys.stdout.write(f"\r  Trial {trial_count}/{total_trials}:  --- THINK: {word:^10s} ---  ")
        sys.stdout.flush()
        time.sleep(think_duration)

        collection_active = False
        trial_data = eeg_buffer.copy()

        # Ensure consistent trial length
        if trial_data.shape[1] >= trial_samples:
            trial_data = trial_data[:, :trial_samples]
        else:
            # Pad with zeros if not enough data
            pad = np.zeros((n_channels, trial_samples - trial_data.shape[1]))
            trial_data = np.concatenate([trial_data, pad], axis=1)

        all_trials.append((trial_data, word_idx))

        # Rest
        sys.stdout.write(f"\r  Trial {trial_count}/{total_trials}:        rest...              ")
        sys.stdout.flush()
        time.sleep(rest_duration)

    print(f"\n\n  Recording complete: {len(all_trials)} trials")

    # Stop headset
    headset.stop_streaming()
    headset.disconnect()

    # --- Phase 3: Save data ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_path / f"baseline_{headset_type}_{timestamp}.npz"

    eeg_array = np.stack([t[0] for t in all_trials])  # (n_trials, n_ch, n_samples)
    labels_array = np.array([t[1] for t in all_trials])  # (n_trials,)

    np.savez(
        filename,
        eeg=eeg_array,
        labels=labels_array,
        calibration=calibration_data,
        words=np.array(words),
        sfreq=sfreq,
        channels=np.array(montage["channels"]),
        headset=headset_type,
        think_duration=think_duration,
        timestamp=timestamp,
    )

    print(f"\n  Data saved to: {filename}")
    print(f"  Shape: {eeg_array.shape} (trials x channels x samples)")
    print(f"  Words: {dict(zip(words, [int((labels_array == i).sum()) for i in range(len(words))]))}")

    # Quick sanity check
    print("\n  --- Quick Stats ---")
    for i, word in enumerate(words):
        word_trials = eeg_array[labels_array == i]
        amp_mean = np.abs(word_trials).mean()
        amp_std = np.abs(word_trials).std()
        print(f"  {word:>10s}: mean_amp={amp_mean:.1f} uV, std={amp_std:.1f} uV, trials={len(word_trials)}")

    return filename


def _create_headset(headset_type: str):
    """Create the appropriate headset."""
    if headset_type == "simulated":
        from reverse_bci.headsets.simulated import SimulatedHeadset
        headset = SimulatedHeadset(headset_type="muse", embed_signal=True)
        headset.connect()
        return headset
    elif headset_type == "muse":
        from reverse_bci.headsets.muse import MuseHeadset
        return MuseHeadset()
    elif headset_type.startswith("emotiv"):
        from reverse_bci.headsets.emotiv import EmotivHeadset
        model = headset_type.replace("emotiv_", "")
        return EmotivHeadset(model=model)
    else:
        raise ValueError(f"Unknown headset: {headset_type}")


def main():
    parser = argparse.ArgumentParser(description="Record EEG baseline data")
    parser.add_argument("--headset", default="simulated",
                        choices=["simulated", "muse", "emotiv_epoc", "emotiv_insight"])
    parser.add_argument("--words", default="yes,no,water,pain,help",
                        help="Comma-separated list of target words")
    parser.add_argument("--trials", type=int, default=100,
                        help="Trials per word")
    parser.add_argument("--think-time", type=float, default=4.0,
                        help="Thinking duration per trial (seconds)")
    parser.add_argument("--output", default="./eeg_recordings",
                        help="Output directory")
    parser.add_argument("--calibration-time", type=float, default=60.0,
                        help="Resting-state calibration duration (seconds)")

    args = parser.parse_args()

    run_recording(
        headset_type=args.headset,
        words=args.words.split(","),
        trials_per_word=args.trials,
        think_duration=args.think_time,
        calibration_duration=args.calibration_time,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
