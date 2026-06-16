"""Extract logits for finetuned CLAP model across all 5 folds."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import laion_clap

from finetune_clap import CLAPFinetuneModel, FinetuneCLAPDataset, CLAPAudioPreprocessor
from models import BaseClassifier

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    metadata_csv = "data/processed_dataset.csv"
    audio_dir = "data/BSD10k-v1.2/audio"
    text_emb_dir = "data/BSD10k-v1.2/features/clap_text_embeddings"
    output_dir = "model_output_finetune/both"
    
    full_df = pd.read_csv(metadata_csv)
    preprocessor = CLAPAudioPreprocessor()
    
    # Reload CLAP module once to avoid reloading time
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    from huggingface_hub import hf_hub_download
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module.load_ckpt(clap_ckpt)
    
    for fold in range(5):
        print(f"\nProcessing Fold {fold}...")
        fold_dir = os.path.join(output_dir, f"fold_{fold}")
        ckpt_path = os.path.join(fold_dir, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"Skipping fold {fold}, no checkpoint found.")
            continue
            
        # Get test split (same splits logic as rest of codebase)
        splits_path = f"model_output/both/fold_{fold}/splits.csv"
        splits = pd.read_csv(splits_path)
        test_indices = splits[splits['split'] == 'test']['index'].astype(str).tolist()
        test_df = full_df[full_df['index'].astype(str).isin(test_indices)].sort_values('index').reset_index(drop=True)
        
        test_dataset = FinetuneCLAPDataset(test_df, audio_dir, text_emb_dir, preprocessor, training=False)
        _nw = 0 if torch.cuda.is_available() else 4
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=_nw)
        
        # Load model
        hatr = BaseClassifier(
            hidden_size=128, num_classes=23,
            emb_size_audio=512, emb_size_text=512,
            dropout=0.1, use_batch_norm=True, mode='both'
        )
        model = CLAPFinetuneModel(clap_module, hatr, freeze_early_layers=True)
        
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.clap_model.audio_branch.load_state_dict(ckpt['clap_audio_branch'])
        model.clap_model.audio_projection.load_state_dict(ckpt['clap_audio_projection'])
        model.hatr.load_state_dict(ckpt['hatr'])
        model = model.to(device)
        model.eval()
        
        logits_list = []
        labels_list = []
        ids_list = []
        
        with torch.no_grad():
            for data in tqdm(test_loader, desc=f"Fold {fold} test inference"):
                mel_fusion = data['mel_fusion'].to(device)
                waveform = data['waveform'].to(device)
                text_emb = data['text_embedding'].to(device)
                
                _, class_logits, _, _, _ = model(mel_fusion, waveform, text_emb)
                logits_list.append(class_logits.cpu().numpy())
                labels_list.append(data['class_idx'].numpy())
                
                # 'sound_id' might be a tensor if dataset yields numeric index or list of strings
                sound_id = data['sound_id']
                if isinstance(sound_id, torch.Tensor):
                    ids_list.extend(sound_id.tolist())
                elif isinstance(sound_id, tuple) or isinstance(sound_id, list):
                    ids_list.extend(list(sound_id))
                else:
                    ids_list.append(sound_id)
        
        all_logits = np.concatenate(logits_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_ids = np.array(ids_list, dtype=object)
        
        np.save(os.path.join(fold_dir, "test_logits.npy"), all_logits)
        np.save(os.path.join(fold_dir, "test_labels.npy"), all_labels)
        np.save(os.path.join(fold_dir, "test_ids.npy"), all_ids)
        
        # Verify accuracy
        preds = all_logits.argmax(axis=1)
        acc = (preds == all_labels).mean()
        print(f"Fold {fold} accuracy: {acc*100:.2f}%")
        
        # Free memory
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
