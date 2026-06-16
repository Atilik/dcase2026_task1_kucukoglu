"""Extract MATPAC audio embeddings for BSD10k dataset.
MATPAC is a SOTA SSL audio encoder outputting 3840-dim embeddings.
Requires: pip install -e ./matpac_model/inference_matpac
Checkpoint: matpac_10_2048.pt (download from GitHub releases)
"""
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

SAMPLE_RATE = 16000  # MATPAC requires 16kHz

def load_model(checkpoint_path, device):
    """Load MATPAC model for embedding extraction."""
    from matpac.model import get_matpac
    model = get_matpac(checkpoint_path=checkpoint_path)
    model.to(device)
    model.eval()
    return model


def extract_embedding(model, audio_path, device):
    """Extract mean-pooled MATPAC embedding [3840] from audio file."""
    waveform, sr = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    # Pad very short audio to minimum 1 second
    min_samples = SAMPLE_RATE  # 1 second
    if waveform.shape[1] < min_samples:
        pad_size = min_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))

    waveform = waveform.to(device)

    with torch.no_grad():
        # emb shape: (batch, 3840), layer_results: (batch, 12, 3840)
        emb, layer_results = model(waveform)

    return emb.squeeze(0).cpu().numpy()  # [3840]


def main():
    parser = argparse.ArgumentParser(description="Extract MATPAC embeddings")
    parser.add_argument("--audio_dir", type=str, default="data/BSD10k-v1.2/audio",
                        help="Directory containing audio files")
    parser.add_argument("--output_dir", type=str, default="data/BSD10k-v1.2/features/matpac_audio_embeddings",
                        help="Output directory for embeddings")
    parser.add_argument("--checkpoint", type=str, default="matpac_model/matpac_10_2048.pt",
                        help="Path to MATPAC checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Audio dir: {args.audio_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Checkpoint: {args.checkpoint}")

    model = load_model(args.checkpoint, device)
    print("MATPAC model loaded successfully")

    os.makedirs(args.output_dir, exist_ok=True)

    audio_files = sorted([f for f in os.listdir(args.audio_dir) if f.endswith(('.wav', '.flac', '.mp3'))])
    print(f"\n=== Extracting MATPAC embeddings: {len(audio_files)} files ===")

    skipped = 0
    errors = 0
    for fname in tqdm(audio_files, desc="MATPAC"):
        out_path = os.path.join(args.output_dir, fname.rsplit('.', 1)[0] + '.npy')

        # Skip if already extracted
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            audio_path = os.path.join(args.audio_dir, fname)
            emb = extract_embedding(model, audio_path, device)
            np.save(out_path, emb)
        except Exception as e:
            print(f"  Error on {fname}: {e}")
            errors += 1

    # Verify
    sample_files = [f for f in os.listdir(args.output_dir) if f.endswith('.npy')]
    if sample_files:
        sample = np.load(os.path.join(args.output_dir, sample_files[0]))
        print(f"\nDone! Shape: {sample.shape}, skipped: {skipped}, errors: {errors}")
        print(f"Total embeddings: {len(sample_files)}")


if __name__ == "__main__":
    main()
