"""Extract Whisper encoder embeddings for audio files.
Mean-pools encoder output to get a single [512] vector per audio file.
Uses whisper-base (512-dim encoder).

Usage:
  python extract_whisper_embeddings.py  # default BSD10k
  python extract_whisper_embeddings.py --audio-dir data/BSD35k-CS/audio --output-dir data/BSD35k-CS/features/whisper_audio_embeddings
"""
import os
import sys
import argparse
import numpy as np
import torch
import soundfile as sf
from pathlib import Path


MODEL_NAME = "openai/whisper-base"
TARGET_SR = 16000


def resample_audio(audio, orig_sr, target_sr):
    """Simple linear interpolation resampling."""
    if orig_sr == target_sr:
        return audio
    ratio = target_sr / orig_sr
    new_length = int(len(audio) * ratio)
    indices = np.linspace(0, len(audio) - 1, new_length)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio-dir", default="data/BSD10k-v1.2/audio")
    parser.add_argument("--output-dir", default="data/BSD10k-v1.2/features/whisper_audio_embeddings")
    args = parser.parse_args()
    
    AUDIO_DIR = args.audio_dir
    OUTPUT_DIR = args.output_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Audio dir: {AUDIO_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")

    # Load model
    from transformers import WhisperModel, WhisperFeatureExtractor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME).encoder.to(device)
    model.eval()
    print(f"Loaded {MODEL_NAME} encoder")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(('.wav', '.flac', '.mp3'))])
    print(f"Found {len(audio_files)} audio files")

    errors = 0
    skipped = 0

    for i, fname in enumerate(audio_files):
        sound_id = fname.rsplit('.', 1)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{sound_id}.npy")

        # Resume support
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            audio_path = os.path.join(AUDIO_DIR, fname)
            audio, sr = sf.read(audio_path, dtype="float32")

            # Convert stereo to mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            # Resample to 16kHz
            if sr != TARGET_SR:
                audio = resample_audio(audio, sr, TARGET_SR)

            # Truncate to 30s (Whisper max)
            max_samples = 30 * TARGET_SR
            if len(audio) > max_samples:
                audio = audio[:max_samples]

            # Extract features
            inputs = feature_extractor(audio, sampling_rate=TARGET_SR, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            with torch.no_grad():
                encoder_output = model(input_features).last_hidden_state  # (1, T, 512)
                embedding = encoder_output.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)  # (512,)

            np.save(out_path, embedding)

            if (i + 1) % 500 == 0:
                print(f"  [{i+1}/{len(audio_files)}] Processed, shape={embedding.shape}")

        except Exception as e:
            errors += 1
            print(f"  Error on {fname}: {e}")

    # Verify
    total = len(audio_files)
    extracted = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')])
    sample = np.load(os.path.join(OUTPUT_DIR, audio_files[0].rsplit('.', 1)[0] + '.npy'))
    print(f"\nDone! Total={total}, Extracted={extracted}, Skipped={skipped}, Errors={errors}")
    print(f"Sample embedding shape: {sample.shape}")


if __name__ == "__main__":
    main()
