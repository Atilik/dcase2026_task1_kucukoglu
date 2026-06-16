"""
Concatenate CLAP audio (512-dim) + BEATs audio (768-dim) embeddings into 1280-dim vectors.
Only creates concatenated embeddings for files that exist in BOTH folders.
"""
import os
import glob
import numpy as np
from tqdm import tqdm

CLAP_DIR = "data/BSD10k-v1.2/features/clap_audio_embeddings"
BEATS_DIR = "data/BSD10k-v1.2/features/beats_audio_embeddings"
OUTPUT_DIR = "data/BSD10k-v1.2/features/concat_audio_embeddings"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find files present in both folders
clap_ids = {os.path.splitext(f)[0] for f in os.listdir(CLAP_DIR) if f.endswith('.npy')}
beats_ids = {os.path.splitext(f)[0] for f in os.listdir(BEATS_DIR) if f.endswith('.npy')}
common_ids = sorted(clap_ids & beats_ids)

print(f"CLAP audio embeddings: {len(clap_ids)}")
print(f"BEATs audio embeddings: {len(beats_ids)}")
print(f"Common (will concatenate): {len(common_ids)}")
print(f"Output dir: {OUTPUT_DIR}")

# Skip already done
done = {os.path.splitext(f)[0] for f in os.listdir(OUTPUT_DIR) if f.endswith('.npy')}
todo = [sid for sid in common_ids if sid not in done]
print(f"Already done: {len(done)}, remaining: {len(todo)}")

for sound_id in tqdm(todo, desc="Concatenating"):
    clap_emb = np.load(os.path.join(CLAP_DIR, f"{sound_id}.npy"))
    beats_emb = np.load(os.path.join(BEATS_DIR, f"{sound_id}.npy"))
    concat_emb = np.concatenate([clap_emb, beats_emb]).astype(np.float32)
    np.save(os.path.join(OUTPUT_DIR, f"{sound_id}.npy"), concat_emb)

total = len(os.listdir(OUTPUT_DIR))
# Verify one
sample = np.load(os.path.join(OUTPUT_DIR, f"{common_ids[0]}.npy"))
print(f"\nDone! {total} concatenated embeddings saved.")
print(f"Sample shape: {sample.shape}, dtype: {sample.dtype}")
