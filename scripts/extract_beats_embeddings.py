import os
import sys
import glob
import numpy as np
import torch
import soundfile as sf
import torchaudio
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "beats_model"))
from BEATs import BEATs, BEATsConfig

CHECKPOINT_PATH = "beats_model/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"
AUDIO_DIR = "data/BSD10k-v1.2/audio"
OUTPUT_DIR = "data/BSD10k-v1.2/features/beats_audio_embeddings"
TARGET_SR = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading BEATs checkpoint...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
cfg = BEATsConfig(checkpoint["cfg"])
model = BEATs(cfg)
model.load_state_dict(checkpoint["model"])
model = model.to(device)
model.eval()
print("BEATs model loaded.")

audio_files = sorted(glob.glob(os.path.join(AUDIO_DIR, "*.wav")))
print(f"Found {len(audio_files)} audio files.")

done = set(os.listdir(OUTPUT_DIR))
audio_files = [f for f in audio_files if os.path.basename(f).replace(".wav", ".npy") not in done]
print(f"Remaining to process: {len(audio_files)}")

def load_audio(filepath):
    waveform, sr = sf.read(filepath, dtype="float32")
    waveform = torch.from_numpy(waveform)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)
    if sr != TARGET_SR:
        waveform = waveform.unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        waveform = waveform.squeeze(0)
    max_samples = TARGET_SR * 30
    if waveform.shape[0] > max_samples:
        waveform = waveform[:max_samples]
    return waveform

print("Extracting embeddings...")
with torch.no_grad():
    for filepath in tqdm(audio_files):
        sound_id = os.path.basename(filepath).replace(".wav", "")
        output_path = os.path.join(OUTPUT_DIR, f"{sound_id}.npy")
        try:
            waveform = load_audio(filepath)
            waveform = waveform.unsqueeze(0).to(device)
            x, _, _ = model.extract_features(waveform)
            embedding = x.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(output_path, embedding)
        except Exception as e:
            print(f"Error processing {sound_id}: {e}")
            continue

num_done = len(os.listdir(OUTPUT_DIR))
print(f"\nDone! Extracted {num_done} embeddings to {OUTPUT_DIR}")
