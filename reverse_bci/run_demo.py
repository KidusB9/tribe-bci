#!/usr/bin/env python3
"""
Reverse BCI Demo - End-to-end demonstration.

Runs through the full pipeline with a simulated headset to show
how the system works. No real EEG hardware needed.

Usage:
    python -m reverse_bci.run_demo
    python -m reverse_bci.run_demo --train    # Train models first
    python -m reverse_bci.run_demo --live      # Live decoding demo
"""

import sys
import time
import logging
import argparse
import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("reverse_bci.demo")


def demo_signal_processing():
    """Demonstrate the EEG signal processing pipeline."""
    print("\n" + "=" * 60)
    print("  DEMO 1: EEG Signal Processing")
    print("=" * 60)

    from reverse_bci.eeg_processor import EEGProcessor, EEGProcessorConfig

    config = EEGProcessorConfig(headset="muse", window_seconds=2.0)
    processor = EEGProcessor(config)

    print(f"  Headset: Muse ({processor.n_channels} channels, {processor.sfreq} Hz)")
    print(f"  Window: {config.window_seconds}s = {processor.window_samples} samples")
    print(f"  Feature dim: {processor.feature_dim}")

    # Generate synthetic EEG
    duration_sec = 4.0
    n_samples = int(duration_sec * processor.sfreq)
    t = np.arange(n_samples) / processor.sfreq

    # Simulate realistic EEG
    eeg = np.random.randn(4, n_samples) * 2.0  # Background noise
    eeg[0] += 5.0 * np.sin(2 * np.pi * 10 * t)  # Alpha rhythm on TP9
    eeg[3] += 5.0 * np.sin(2 * np.pi * 10 * t)  # Alpha on TP10
    eeg[1] += 0.5 * np.sin(2 * np.pi * 60 * t)  # Line noise on AF7
    # Add eye blink on frontal channels
    blink_center = int(1.5 * processor.sfreq)
    blink = 50 * np.exp(-((np.arange(n_samples) - blink_center) ** 2) / (0.01 * processor.sfreq ** 2))
    eeg[1] += blink
    eeg[2] += blink

    processor.push_chunk(eeg)
    print(f"\n  Pushed {n_samples} samples ({duration_sec}s)")

    windows = processor.process_all()
    print(f"  Processed {len(windows)} windows")

    for i, features in enumerate(windows):
        quality = features["quality"]
        band_powers = features["band_powers"]
        print(f"\n  Window {i + 1}:")
        print(f"    Signal quality: {quality['overall']:.2f}")
        print(f"    Clean ratio:    {quality['clean_ratio']:.2f}")
        print(f"    Mean SNR:       {quality['mean_snr_db']:.1f} dB")
        print(f"    Band powers:")
        for band, powers in sorted(band_powers.items()):
            print(f"      {band:>6s}: {powers.mean():.3f} (log-power)")

    print("\n  Signal processing: OK")


def demo_neural_pipeline():
    """Demonstrate the neural network pipeline."""
    print("\n" + "=" * 60)
    print("  DEMO 2: Neural Network Pipeline")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # EEG Encoder
    from reverse_bci.eeg_encoder import EEGEncoder, EEGEncoderConfig
    encoder_config = EEGEncoderConfig(n_channels=4, n_samples=512, latent_dim=1152)
    encoder = EEGEncoder(encoder_config).to(device)
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"\n  EEG Encoder: {n_params:,} parameters")

    # Domain Adapter
    from reverse_bci.domain_adapter import DomainAdapter, DomainAdapterConfig
    adapter = DomainAdapter(DomainAdapterConfig()).to(device)
    n_params = sum(p.numel() for p in adapter.parameters())
    print(f"  Domain Adapter: {n_params:,} parameters")

    # Reverse Decoder
    from reverse_bci.reverse_decoder import ReverseDecoder, ReverseDecoderConfig
    decoder = ReverseDecoder(ReverseDecoderConfig()).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Reverse Decoder: {n_params:,} parameters")

    # Text Decoder
    from reverse_bci.text_decoder import TextDecoder, TextDecoderConfig
    text_decoder = TextDecoder(TextDecoderConfig()).to(device)
    n_params = sum(p.numel() for p in text_decoder.parameters())
    print(f"  Text Decoder: {n_params:,} parameters")
    print(f"  Vocabulary: {text_decoder.get_vocab_size()} words")

    # Forward pass
    print("\n  Running forward pass...")
    eeg_input = torch.randn(1, 4, 512).to(device)

    with torch.no_grad():
        t0 = time.time()

        # Step 1: EEG -> Latent
        enc_out = encoder(eeg_input, return_attention=True)
        eeg_latent = enc_out["latent"]
        print(f"    EEG -> Latent: {eeg_latent.shape}")

        if "attention" in enc_out:
            attn = enc_out["attention"][0]
            channels = ["TP9", "AF7", "AF8", "TP10"]
            print(f"    Spatial attention: {dict(zip(channels, [f'{a:.2f}' for a in attn.tolist()]))}")

        # Step 2: Domain adaptation
        tribe_latent = adapter(eeg_latent)
        print(f"    Adapted latent: {tribe_latent.shape}")

        # Step 3: Reverse decode -> text features
        dec_out = decoder(tribe_latent)
        text_features = dec_out["text_features"]
        print(f"    Text features: {text_features.shape}")

        # Step 4: Decode to words
        text_out = text_decoder(text_features)
        t1 = time.time()

        print(f"\n  Decoded word: '{text_out['decoded_words'][0]}'")
        conf = text_out["confidence"][0].item()
        print(f"  Confidence: {conf:.2%}")
        print(f"  Top alternatives:")
        for word, score in text_out["alternatives"][0][:5]:
            print(f"    {word:>15s}: {score:.4f}")
        print(f"\n  Total latency: {(t1 - t0) * 1000:.1f} ms")

    print("\n  Neural pipeline: OK")


def demo_simulated_session():
    """Run a short simulated BCI session."""
    print("\n" + "=" * 60)
    print("  DEMO 3: Simulated BCI Session")
    print("=" * 60)

    from reverse_bci.bci_pipeline import BCIPipeline, BCIConfig
    from reverse_bci.headsets.simulated import SimulatedHeadset

    config = BCIConfig(
        headset_type="muse",
        confidence_threshold=0.1,  # Low threshold for demo
        calibration_duration=5.0,
        smoothing_window=1,
    )

    pipeline = BCIPipeline(config)
    pipeline.load_models()

    headset = SimulatedHeadset(headset_type="muse", embed_signal=True)
    headset.connect()
    pipeline.connect_headset(headset)

    decoded_words = []

    def on_word(word, conf):
        decoded_words.append((word, conf))
        print(f"    Decoded: {word:>15s} (confidence: {conf:.0%})")

    pipeline.on_word(on_word)

    print("\n  Starting 10-second simulated session...")
    pipeline.start()

    # Run for 10 seconds
    for i in range(20):
        time.sleep(0.5)
        if i % 4 == 0:
            # Change the simulated "thought"
            headset.set_target_word(i // 4)

    pipeline.stop()

    print(f"\n  Session complete:")
    print(f"    Words decoded: {len(decoded_words)}")
    print(f"    Sentence: {pipeline.output_buffer.current_sentence}")
    stats = pipeline.stats
    print(f"    Avg latency: {stats['avg_latency_ms']:.1f} ms")

    print("\n  Simulated session: OK")


def demo_training():
    """Demonstrate the training pipeline."""
    print("\n" + "=" * 60)
    print("  DEMO 4: Training Pipeline")
    print("=" * 60)

    from reverse_bci.training.train_decoder import DecoderTrainer
    from reverse_bci.reverse_decoder import ReverseDecoderConfig
    from reverse_bci.text_decoder import TextDecoderConfig

    print("  Training reverse decoder on synthetic data...")
    print("  (Using small config for demo)")

    trainer = DecoderTrainer(
        decoder_config=ReverseDecoderConfig(),
        text_config=TextDecoderConfig(),
        lr=1e-3,
        batch_size=16,
        n_epochs=5,
    )

    # Quick training with small dataset
    trainer.train_reverse_decoder(tribe_model=None, n_augmentations=5)
    print("\n  Training text decoder...")
    trainer.train_text_decoder(tribe_model=None, n_augmentations=5)

    # Save checkpoint
    trainer.save("./cache/demo_decoder.pt")
    print("\n  Training pipeline: OK")


def main():
    parser = argparse.ArgumentParser(description="Reverse BCI Demo")
    parser.add_argument(
        "--all", action="store_true",
        help="Run all demos",
    )
    parser.add_argument(
        "--signal", action="store_true",
        help="Demo signal processing only",
    )
    parser.add_argument(
        "--neural", action="store_true",
        help="Demo neural pipeline only",
    )
    parser.add_argument(
        "--session", action="store_true",
        help="Demo simulated BCI session",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Demo training pipeline",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run live decoding (with real or simulated headset)",
    )

    args = parser.parse_args()

    # Default: run all demos
    run_all = args.all or not any([args.signal, args.neural, args.session, args.train, args.live])

    print()
    print("*" * 60)
    print("*  REVERSE BCI: Non-invasive Brain-Computer Interface      *")
    print("*  Powered by TRIBE v2 (Meta AI)                           *")
    print("*                                                          *")
    print("*  Decoding thoughts from consumer EEG headsets.            *")
    print("*  No brain surgery required.                               *")
    print("*" * 60)

    if run_all or args.signal:
        demo_signal_processing()

    if run_all or args.neural:
        demo_neural_pipeline()

    if run_all or args.train:
        demo_training()

    if run_all or args.session:
        demo_simulated_session()

    if args.live:
        from reverse_bci.ui.app import BCIApp
        app = BCIApp(headset="simulated")
        app.run()

    print("\n" + "=" * 60)
    print("  All demos completed successfully!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("  1. Connect a real EEG headset (Muse or Emotiv)")
    print("  2. Train the model: python -m reverse_bci.run_demo --train")
    print("  3. Run live decoding: python -m reverse_bci.run_demo --live")
    print("  4. Full training with TRIBE v2:")
    print("     python -m reverse_bci.train_full --tribe facebook/tribev2")
    print()


if __name__ == "__main__":
    main()
