import os
import glob
import pandas as pd
import numpy as np
import torch
import torchaudio
import soundfile as sf
import librosa
from tqdm import tqdm
from huggingface_hub import hf_hub_download
import laion_clap

from external_label_mappings import BST_SYNTHETIC_TEXT

# --- Config ---
EXTERNAL_MAPPING = "data/external_mapping.csv"
AUDIO_OUT_DIR = "data/external_embeddings/clap_audio_embeddings"
TEXT_OUT_DIR = "data/external_embeddings/clap_text_embeddings"
SAMPLE_RATE = 48000

def load_clap_model(device):
    print("Loading LAION-CLAP model...")
    # Using HTSAT-tiny and 630k-audioset-fusion-best.pt as requested,
    # or 630k-audioset-best.pt if fusion is false. The prompt said fusion-best.
    # We'll use enable_fusion=True if fusion-best is requested.
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module.load_ckpt(clap_ckpt)
    clap_module.to(device)
    clap_module.eval()
    return clap_module


def process_audio(audio_path, target_sr=48000):
    # Load with soundfile, which handles wav, flac, ogg
    audio_data, sr = sf.read(audio_path)
    
    # Convert to mono if necessary
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    # Resample if necessary
    if sr != target_sr:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
    # LAION-CLAP requires int16 or float32. We'll use float32 in range [-1.0, 1.0]
    audio_data = audio_data.astype(np.float32)
    
    # LAION-CLAP get_audio_embedding_from_data expects a list of float32 arrays
    # or a tensor. We'll pass it a list with one item or a tensor.
    # It requires the input to be shape (1, num_samples). Let's reshape/pad.
    
    # If the audio is very long, CLAP handles it if enable_fusion=True.
    # LAION CLAP expects audio to be padded or truncated to exactly 10 seconds?
    # Actually, get_audio_embedding_from_data handles it by default (pads/truncates internally).
    
    # Ensure it's 1D for get_audio_embedding_from_data
    return audio_data


def extract_audio_embeddings(clap_module, df, device):
    os.makedirs(AUDIO_OUT_DIR, exist_ok=True)
    
    print("Extracting audio embeddings...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_id = row['source_id']
        audio_path = row['audio_path']
        out_path = os.path.join(AUDIO_OUT_DIR, f"{source_id}.npy")
        
        # Skip if already exists
        if os.path.exists(out_path):
            continue
            
        try:
            audio_data = process_audio(audio_path, SAMPLE_RATE)
            
            # The get_audio_embedding_from_data takes a list of audio arrays
            # or a tensor of shape (batch, samples). Use_tensor=False passes list
            with torch.no_grad():
                # use_tensor=False means input is list of numpy arrays
                embed = clap_module.get_audio_embedding_from_data(x=[audio_data], use_tensor=False)
                
            # embed is shape (1, 512). Extract and save.
            embed = embed[0]
            np.save(out_path, embed)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")


def extract_text_embeddings(clap_module, df, device):
    os.makedirs(TEXT_OUT_DIR, exist_ok=True)
    
    print("Pre-computing text embeddings for BST classes...")
    # First, get unique BST classes
    bst_classes = df['bst_class'].unique()
    
    # Compute embeddings for each class
    class_embeddings = {}
    for bst_class in bst_classes:
        if bst_class not in BST_SYNTHETIC_TEXT:
            print(f"WARNING: No text description for {bst_class}")
            # Use the class name itself as fallback
            text = bst_class
        else:
            text = BST_SYNTHETIC_TEXT[bst_class]
            
        with torch.no_grad():
            embed = clap_module.get_text_embedding([text], use_tensor=False)
        class_embeddings[bst_class] = embed[0]
        
    print("Saving text embeddings for all samples...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        source_id = row['source_id']
        bst_class = row['bst_class']
        out_path = os.path.join(TEXT_OUT_DIR, f"{source_id}.npy")
        
        # Skip if already exists
        if os.path.exists(out_path):
            continue
            
        if bst_class in class_embeddings:
            np.save(out_path, class_embeddings[bst_class])
        else:
            print(f"No embedding found for {bst_class}")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    df = pd.read_csv(EXTERNAL_MAPPING)
    print(f"Loaded {len(df)} mapped external samples.")
    
    clap_module = load_clap_model(device)
    
    extract_text_embeddings(clap_module, df, device)
    extract_audio_embeddings(clap_module, df, device)
    
    print("Done!")

if __name__ == "__main__":
    main()
