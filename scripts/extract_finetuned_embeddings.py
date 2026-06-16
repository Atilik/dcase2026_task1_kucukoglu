"""Extract audio embeddings from fine-tuned CLAP model.

Loads the best fine-tuned CLAP audio encoder (fold 0 by default, or ensemble-averaged)
and extracts 512-dim embeddings for all BSD10k audio files.

Output: data/BSD10k-v1.2/features/finetuned_clap_audio_embeddings/{sound_id}.npy

Usage:
    # GPU required
    python -u extract_finetuned_embeddings.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm

from finetune_clap import CLAPFinetuneModel, CLAPAudioPreprocessor
from models import BaseClassifier

# Config
METADATA_CSV = "data/processed_dataset.csv"
AUDIO_DIR = "data/BSD10k-v1.2/audio"
FINETUNE_DIR = "model_output_finetune/both"
OUTPUT_DIR = "data/BSD10k-v1.2/features/finetuned_clap_audio_embeddings"
CLAP_SR = 48000
MAX_DURATION = 10
MAX_SAMPLES = CLAP_SR * MAX_DURATION
NUM_FOLDS = 5
BATCH_SIZE = 32


def load_audio(audio_path, target_sr=48000, max_samples=480000):
    """Load and preprocess a single audio file."""
    try:
        waveform, sr = sf.read(audio_path, dtype='float32')
    except Exception:
        # Try .flac extension
        audio_path = audio_path.replace('.wav', '.flac')
        waveform, sr = sf.read(audio_path, dtype='float32')

    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)

    # Resample to 48kHz if needed
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            torch.tensor(waveform), orig_freq=sr, new_freq=target_sr
        ).numpy()

    # Pad or truncate to max_samples
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]  # deterministic crop for extraction
    else:
        waveform = np.pad(waveform, (0, max_samples - len(waveform)))

    return torch.tensor(waveform, dtype=torch.float32)


def load_finetuned_model(clap_module, fold, device):
    """Load fine-tuned CLAP model for a specific fold."""
    fold_dir = os.path.join(FINETUNE_DIR, f"fold_{fold}")
    ckpt_path = os.path.join(fold_dir, "best_model.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")

    hatr = BaseClassifier(
        hidden_size=128, num_classes=23,
        emb_size_audio=512, emb_size_text=512,
        dropout=0.1, use_batch_norm=True, mode='both'
    )
    model = CLAPFinetuneModel(clap_module, hatr, freeze_early_layers=False)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.clap_model.audio_branch.load_state_dict(ckpt['clap_audio_branch'])
    model.clap_model.audio_projection.load_state_dict(ckpt['clap_audio_projection'])
    model = model.to(device)
    model.eval()

    print(f"  Loaded fold {fold} checkpoint (val_acc={ckpt.get('val_acc', '?'):.2f}%)")
    return model


def extract_embedding(model, waveform, preprocessor, device):
    """Extract 512-dim audio embedding from fine-tuned CLAP."""
    fbank = preprocessor(waveform)  # (time, mel_bins)

    with torch.no_grad():
        # Build mel_fusion: (1, 4, time, mel_bins) — CLAP fusion format
        mel_fusion = fbank.unsqueeze(0).repeat(4, 1, 1)  # (4, time, mel_bins)
        mel_fusion[1:, :, :] = 0.0  # zero channels 1-3 for non-longer clips
        mel_fusion = mel_fusion.unsqueeze(0).to(device)  # (1, 4, time, mel_bins)
        waveform = waveform.unsqueeze(0).to(device)  # (1, samples)

        input_dict = {
            'mel_fusion': mel_fusion,
            'waveform': waveform,
            'longer': torch.tensor([False]).to(device),
        }
        audio_output = model.clap_model.audio_branch(input_dict, mixup_lambda=None)
        audio_emb = model.clap_model.audio_projection(audio_output["embedding"])

    return audio_emb.squeeze(0).cpu().numpy()  # (512,)


def extract_embeddings_batched(model, sound_ids, preprocessor, device):
    """Extract embeddings in batches for efficiency."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find which files still need processing
    to_process = []
    for sid in sound_ids:
        out_path = os.path.join(OUTPUT_DIR, f"{sid}.npy")
        if not os.path.exists(out_path):
            to_process.append(sid)

    if not to_process:
        print(f"  All {len(sound_ids)} embeddings already exist, skipping.")
        return

    print(f"  Extracting {len(to_process)} embeddings ({len(sound_ids) - len(to_process)} cached)...")

    for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Extracting"):
        batch_ids = to_process[i:i + BATCH_SIZE]
        mel_fusions = []
        waveforms = []
        valid_ids = []

        for sid in batch_ids:
            audio_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
            try:
                waveform = load_audio(audio_path, CLAP_SR, MAX_SAMPLES)
                fbank = preprocessor(waveform)
                # Build mel_fusion: (4, time, mel_bins)
                mel_fusion = fbank.unsqueeze(0).repeat(4, 1, 1)
                mel_fusion[1:, :, :] = 0.0
                mel_fusions.append(mel_fusion)
                waveforms.append(waveform)
                valid_ids.append(sid)
            except Exception as e:
                print(f"  Error loading {sid}: {e}")

        if not valid_ids:
            continue

        with torch.no_grad():
            mel_fusion_batch = torch.stack(mel_fusions).to(device)
            waveform_batch = torch.stack(waveforms).to(device)

            input_dict = {
                'mel_fusion': mel_fusion_batch,
                'waveform': waveform_batch,
                'longer': torch.tensor([False] * mel_fusion_batch.shape[0]).to(device),
            }
            audio_output = model.clap_model.audio_branch(input_dict, mixup_lambda=None)
            audio_embs = model.clap_model.audio_projection(audio_output["embedding"])
            audio_embs = audio_embs.cpu().numpy()

        for sid, emb in zip(valid_ids, audio_embs):
            out_path = os.path.join(OUTPUT_DIR, f"{sid}.npy")
            np.save(out_path, emb)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not torch.cuda.is_available():
        print("WARNING: No GPU! This will be very slow.")

    # Load metadata
    full_df = pd.read_csv(METADATA_CSV)
    sound_ids = full_df['index'].astype(str).tolist()
    print(f"Total samples to extract: {len(sound_ids)}")

    # Load CLAP base model
    print("Loading CLAP model...")
    import laion_clap
    from huggingface_hub import hf_hub_download
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_module.load_ckpt(clap_ckpt)
    print(f"CLAP loaded from {clap_ckpt}")

    preprocessor = CLAPAudioPreprocessor(sample_rate=CLAP_SR, enable_fusion=True)

    # Per-fold extraction: each fold's checkpoint extracts embeddings for ALL samples
    # This ensures no data leakage when the downstream HATR trains with 5-fold CV
    for fold in range(NUM_FOLDS):
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        print(f"\n{'='*60}")
        print(f"FOLD {fold}: Extracting with fold {fold}'s fine-tuned checkpoint")
        print(f"  Output: {fold_output_dir}")
        print(f"{'='*60}")

        # Reload CLAP base weights before loading fold checkpoint
        # (since load_finetuned_model modifies audio_branch in place)
        clap_module.load_ckpt(clap_ckpt)
        model = load_finetuned_model(clap_module, fold, device)

        # Override output dir for this fold
        orig_output = globals().get('_orig_output_dir', OUTPUT_DIR)
        os.makedirs(fold_output_dir, exist_ok=True)

        # Extract all embeddings with this fold's model
        to_process = []
        for sid in sound_ids:
            out_path = os.path.join(fold_output_dir, f"{sid}.npy")
            if not os.path.exists(out_path):
                to_process.append(sid)

        if not to_process:
            print(f"  All {len(sound_ids)} embeddings already exist, skipping.")
            continue

        print(f"  Extracting {len(to_process)} embeddings ({len(sound_ids) - len(to_process)} cached)...")

        for i in tqdm(range(0, len(to_process), BATCH_SIZE), desc="Extracting"):
            batch_ids = to_process[i:i + BATCH_SIZE]
            mel_fusions = []
            waveforms = []
            valid_ids = []

            for sid in batch_ids:
                audio_path = os.path.join(AUDIO_DIR, f"{sid}.wav")
                try:
                    waveform = load_audio(audio_path, CLAP_SR, MAX_SAMPLES)
                    fbank = preprocessor(waveform)
                    mel_fusion = fbank.unsqueeze(0).repeat(4, 1, 1)
                    mel_fusion[1:, :, :] = 0.0
                    mel_fusions.append(mel_fusion)
                    waveforms.append(waveform)
                    valid_ids.append(sid)
                except Exception as e:
                    print(f"  Error loading {sid}: {e}")

            if not valid_ids:
                continue

            with torch.no_grad():
                mel_fusion_batch = torch.stack(mel_fusions).to(device)
                waveform_batch = torch.stack(waveforms).to(device)

                input_dict = {
                    'mel_fusion': mel_fusion_batch,
                    'waveform': waveform_batch,
                    'longer': torch.tensor([False] * mel_fusion_batch.shape[0]).to(device),
                }
                audio_output = model.clap_model.audio_branch(input_dict, mixup_lambda=None)
                audio_embs = model.clap_model.audio_projection(audio_output["embedding"])
                audio_embs = audio_embs.cpu().numpy()

            for sid, emb in zip(valid_ids, audio_embs):
                out_path = os.path.join(fold_output_dir, f"{sid}.npy")
                np.save(out_path, emb)

        # Verify
        existing = sum(1 for sid in sound_ids
                       if os.path.exists(os.path.join(fold_output_dir, f"{sid}.npy")))
        print(f"  Done! {existing}/{len(sound_ids)} embeddings saved")

    # Check embedding dimensions from fold 0
    sample_path = os.path.join(OUTPUT_DIR, "fold_0", f"{sound_ids[0]}.npy")
    if os.path.exists(sample_path):
        sample_emb = np.load(sample_path)
        print(f"\nEmbedding shape: {sample_emb.shape}")

    print(f"\nAll {NUM_FOLDS} folds extracted to {OUTPUT_DIR}/fold_*/")


if __name__ == "__main__":
    main()

