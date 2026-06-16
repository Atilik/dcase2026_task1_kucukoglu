"""Extract ConvNeXt audio embeddings for BSD10k dataset.
Mean-pools frame embeddings to get a single [768] vector per audio file.
"""
import os
import sys
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm

# Add convNeXt_model to path for the ConvNeXt class
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'convNeXt_model'))

try:
    from audioset_convnext_inf.pytorch.convnext import ConvNeXt
except ImportError:
    print("audioset_convnext_inf not found. Trying pip install...")
    os.system("pip install git+https://github.com/topel/audioset-convnext-inf@pip-install")
    from audioset_convnext_inf.pytorch.convnext import ConvNeXt


CHECKPOINT = "convNeXt_model/convnext_tiny_465mAP_BL_AC_70kit.pth"
SAMPLE_RATE = 32000


def load_model(checkpoint_path, device):
    model = ConvNeXt(in_chans=1, num_classes=527, use_torchaudio=False)
    model.downsample_layers[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=False)
    model.to(device)
    model.eval()
    return model


def extract_embedding(model, audio_path, device):
    """Extract mean-pooled ConvNeXt embedding [768] from audio file."""
    data, sr = sf.read(audio_path, dtype='float32')
    # Convert to torch tensor [1, samples]
    waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    
    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Pad very short audio to minimum 1 second to avoid kernel size errors
    min_samples = SAMPLE_RATE  # 32000 = 1 second
    if waveform.shape[1] < min_samples:
        pad_size = min_samples - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    
    waveform = waveform.to(device)
    
    with torch.no_grad():
        # Get frame embeddings: [B, C, T, F]
        frame_emb = model.forward_frame_embeddings(waveform)
        # Global average pool over time and frequency → [B, C]
        scene_emb = frame_emb.mean(dim=[2, 3])
    
    return scene_emb.squeeze(0).cpu().numpy()  # [768]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = load_model(CHECKPOINT, device)
    print("Model loaded successfully")
    
    # Test with one file
    test_file = None
    
    datasets = [
        ("BSD10k-v1.2", "data/BSD10k-v1.2/audio", "data/BSD10k-v1.2/features/convnext_audio_embeddings"),
    ]
    
    for name, audio_dir, output_dir in datasets:
        if not os.path.isdir(audio_dir):
            print(f"Skipping {name}: {audio_dir} not found")
            continue
        
        os.makedirs(output_dir, exist_ok=True)
        
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.flac', '.mp3'))])
        print(f"\n=== Extracting {name}: {len(audio_files)} files → {output_dir} ===")
        
        skipped = 0
        for i, fname in enumerate(tqdm(audio_files, desc=name)):
            out_path = os.path.join(output_dir, fname.rsplit('.', 1)[0] + '.npy')
            
            # Skip if already extracted
            if os.path.exists(out_path):
                skipped += 1
                continue
            
            try:
                audio_path = os.path.join(audio_dir, fname)
                emb = extract_embedding(model, audio_path, device)
                np.save(out_path, emb)
            except Exception as e:
                print(f"  Error on {fname}: {e}")
        
        # Verify
        sample = np.load(os.path.join(output_dir, audio_files[0].rsplit('.', 1)[0] + '.npy'))
        print(f"  Done! Shape: {sample.shape}, skipped: {skipped}")


if __name__ == "__main__":
    main()
