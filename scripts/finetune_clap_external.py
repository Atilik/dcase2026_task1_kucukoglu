"""Fine-tune CLAP audio encoder end-to-end with External Data (Strategy B).

Uses LAION-CLAP with the audio encoder trainable.
K-folds on BSD10k only, adds external data to the training fold with weight=0.5.
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import soundfile as sf
import librosa

from models import BaseClassifier
from utils import set_seed
from finetune_clap import CLAPAudioPreprocessor, CLAPFinetuneModel, init_hatr_weights

# ── Config ──
SEED = 1821
K_FOLDS = 5
NUM_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 8
ACCUMULATION_STEPS = 4
CLAP_LR = 1e-5
HATR_LR = 1e-3
WEIGHT_DECAY = 1e-4
CLAP_SR = 48000
MAX_DURATION = 10

# Data paths
METADATA_CSV = "data/combined_dataset_all.csv"
EXTERNAL_MAPPING = "data/external_mapping.csv"
AUDIO_DIR = "data/BSD10k-v1.2/audio"
TEXT_EMB_DIR = "data/BSD10k-v1.2/features/clap_text_embeddings"
EXT_TEXT_DIR = "data/external_embeddings/clap_text_embeddings"
CLASS_DICT_PATH = "data/class_dict.json"
TOP_CLASS_DICT_PATH = "data/top_class_dict.json"
OUTPUT_DIR = "model_outputs/model_output_finetune_ext"

FOLD_ONLY = 0  # Start with fold 0 only to verify pipeline works, as user requested


# ── Dataset ──
class FinetuneCLAPExternalDataset(Dataset):
    def __init__(self, dataframe, ext_audio_map, bsd10k_audio_dir, bsd10k_text_dir, ext_text_dir, preprocessor,
                 target_sr=48000, max_duration=10, training=True):
        self.dataframe = dataframe
        self.ext_audio_map = ext_audio_map
        self.bsd10k_audio_dir = bsd10k_audio_dir
        self.bsd10k_text_dir = bsd10k_text_dir
        self.ext_text_dir = ext_text_dir
        self.preprocessor = preprocessor
        self.target_sr = target_sr
        self.max_samples = target_sr * max_duration
        self.training = training

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        dataset_source = sample.get('dataset_source', 'BSD10k-v1.2')
        sound_id = str(sample['index'])
        
        # Audio and text path resolution
        if dataset_source == 'BSD10k-v1.2':
            audio_path = os.path.join(self.bsd10k_audio_dir, f"{sound_id}.wav")
            if not os.path.exists(audio_path):
                audio_path = os.path.join(self.bsd10k_audio_dir, f"{sound_id}.flac")
            text_path = os.path.join(self.bsd10k_text_dir, f"{sound_id}.npy")
            sample_weight = 1.0
        else:
            audio_path = self.ext_audio_map[sound_id]
            text_path = os.path.join(self.ext_text_dir, f"{sound_id}.npy")
            sample_weight = 0.5  # Downweight noisy external data

        # Load raw audio
        try:
            waveform, sr = sf.read(audio_path, dtype='float32')
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform, sr = np.zeros(self.max_samples, dtype='float32'), self.target_sr

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Resample to 48kHz if needed
        if sr != self.target_sr:
            import torchaudio.functional as F_audio
            waveform = F_audio.resample(torch.tensor(waveform), orig_freq=sr, new_freq=self.target_sr).numpy()

        # Pad or truncate
        if len(waveform) > self.max_samples:
            if self.training:
                start = np.random.randint(0, len(waveform) - self.max_samples)
                waveform = waveform[start:start + self.max_samples]
            else:
                waveform = waveform[:self.max_samples]
        else:
            waveform = np.pad(waveform, (0, self.max_samples - len(waveform)))

        waveform = torch.tensor(waveform, dtype=torch.float32)
        
        # Compute mel spectrogram (not differentiable, just feature extraction)
        fbank = self.preprocessor(waveform)  # (target_length, mel_bins)

        # Load text embedding
        text_emb = torch.tensor(np.load(text_path), dtype=torch.float32)

        return {
            'fbank': fbank,
            'waveform': waveform,  # keep for fusion mode
            'text_embedding': text_emb,
            'class_idx': sample['class_idx'],
            'top_class_idx': sample['top_class_idx'],
            'sound_id': sound_id,
            'weight': sample_weight
        }


def train_one_epoch(model, train_loader, optimizer, device, accumulation_steps=1, epoch=0, num_epochs=30):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()

    # Use reduction='none' to apply sample weights
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01, reduction='none')

    for step, data in enumerate(train_loader):
        fbank = data['fbank'].to(device)
        waveform = data['waveform'].to(device)
        text_emb = data['text_embedding'].to(device)
        labels = data['class_idx'].to(device)
        weights = data['weight'].to(device)
        batch_size = labels.size(0)

        z, class_logit, _, _, _ = model(fbank, waveform, text_emb)

        # Compute weighted loss
        unweighted_loss = criterion(class_logit, labels)
        loss = (unweighted_loss * weights).mean() / accumulation_steps
        
        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * batch_size
        _, predicted = class_logit.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += batch_size

        if (step + 1) % 50 == 0:
            print(f"    [{epoch+1}/{num_epochs}] Step {step+1}/{len(train_loader)}, "
                  f"Loss: {total_loss/total_samples:.4f}, "
                  f"Acc: {100.*total_correct/total_samples:.1f}%", flush=True)

    return total_loss / total_samples, 100. * total_correct / total_samples


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.01)

    for data in val_loader:
        fbank = data['fbank'].to(device)
        waveform = data['waveform'].to(device)
        text_emb = data['text_embedding'].to(device)
        labels = data['class_idx'].to(device)
        batch_size = labels.size(0)

        z, class_logit, _, _, _ = model(fbank, waveform, text_emb)
        loss = criterion(class_logit, labels)

        total_loss += loss.item() * batch_size
        _, predicted = class_logit.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, 100. * total_correct / total_samples


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load metadata ──
    full_df = pd.read_csv(METADATA_CSV)
    ext_map_df = pd.read_csv(EXTERNAL_MAPPING)
    ext_audio_map = dict(zip(ext_map_df['source_id'].astype(str), ext_map_df['audio_path']))
    
    print(f"Dataset: {len(full_df)} total samples")

    with open(CLASS_DICT_PATH) as f:
        class_dict = json.load(f)
    num_classes = len(class_dict)

    # ── Load CLAP model ──
    print("Loading CLAP model...")
    import laion_clap
    from huggingface_hub import hf_hub_download
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_module.load_ckpt(clap_ckpt)
    
    preprocessor = CLAPAudioPreprocessor(sample_rate=CLAP_SR, enable_fusion=True)

    # ── K-fold CV on BSD10k ONLY ──
    # Split primary (BSD10k) and extra (external)
    primary_df = full_df[full_df['dataset_source'] == 'BSD10k-v1.2'].reset_index(drop=True)
    extra_df = full_df[full_df['dataset_source'] != 'BSD10k-v1.2'].reset_index(drop=True)
    
    print(f"Primary (BSD10k): {len(primary_df)} samples")
    print(f"Extra (external): {len(extra_df)} samples")

    labels = primary_df["class_idx"].tolist()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    all_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if FOLD_ONLY is not None and fold != FOLD_ONLY:
            continue
            
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        # Split train/val
        trainval_labels = [labels[i] for i in trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
        train_idx = [trainval_idx[i] for i in train_idx_rel]
        val_idx = [trainval_idx[i] for i in val_idx_rel]

        train_df = primary_df.iloc[train_idx].reset_index(drop=True)
        val_df = primary_df.iloc[val_idx].reset_index(drop=True)
        test_df = primary_df.iloc[test_idx].reset_index(drop=True)
        
        # ADD EXTERNAL DATA TO TRAINING FOLD ONLY
        train_df = pd.concat([train_df, extra_df], ignore_index=True)
        
        print(f"Train: {len(train_df)} (incl external), Val: {len(val_df)}, Test: {len(test_df)}")

        # ── Datasets & Loaders ──
        train_dataset = FinetuneCLAPExternalDataset(
            train_df, ext_audio_map, AUDIO_DIR, TEXT_EMB_DIR, EXT_TEXT_DIR, preprocessor, training=True
        )
        val_dataset = FinetuneCLAPExternalDataset(
            val_df, ext_audio_map, AUDIO_DIR, TEXT_EMB_DIR, EXT_TEXT_DIR, preprocessor, training=False
        )
        test_dataset = FinetuneCLAPExternalDataset(
            test_df, ext_audio_map, AUDIO_DIR, TEXT_EMB_DIR, EXT_TEXT_DIR, preprocessor, training=False
        )

        # num_workers>0 deadlocks inside Singularity with GPU passthrough
        _nw = 0 if torch.cuda.is_available() else 4
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   drop_last=True, num_workers=_nw, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=_nw, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=_nw, pin_memory=True)

        # ── Build model ──
        if fold > 0 or (FOLD_ONLY is not None and FOLD_ONLY > 0):
            clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
            clap_module.load_ckpt(clap_ckpt)

        hatr = BaseClassifier(
            hidden_size=128, num_classes=num_classes,
            emb_size_audio=512, emb_size_text=512,
            dropout=0.1, use_batch_norm=True, mode='both',
        )
        
        model = CLAPFinetuneModel(clap_module, hatr, freeze_early_layers=True)
        init_hatr_weights(model)
        model = model.to(device)

        # ── Optimizer ──
        audio_branch_params = [p for p in model.clap_model.audio_branch.parameters() if p.requires_grad]
        audio_proj_params = list(model.clap_model.audio_projection.parameters())
        hatr_params = list(model.hatr.parameters())

        optimizer = torch.optim.AdamW([
            {'params': audio_branch_params, 'lr': CLAP_LR, 'weight_decay': WEIGHT_DECAY},
            {'params': audio_proj_params, 'lr': CLAP_LR * 5, 'weight_decay': WEIGHT_DECAY},
            {'params': hatr_params, 'lr': HATR_LR, 'weight_decay': 1e-5},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

        # ── Training loop ──
        fold_dir = os.path.join(OUTPUT_DIR, "both", f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0

        # Enable mixed precision scaler to save GPU memory
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(NUM_EPOCHS):
            # For mixed precision training, we'd need to modify train_one_epoch, 
            # but since CLAP is so memory-hungry we just stick to standard train_one_epoch 
            # as it was already implemented in finetune_clap.py without amp.
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, device,
                accumulation_steps=ACCUMULATION_STEPS,
                epoch=epoch, num_epochs=NUM_EPOCHS
            )
            val_loss, val_acc = validate(model, val_loader, device)
            scheduler.step()

            lr_audio = optimizer.param_groups[0]['lr']
            lr_hatr = optimizer.param_groups[2]['lr']
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.2f}% | "
                  f"LR: audio={lr_audio:.2e}", flush=True)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'clap_audio_branch': model.clap_model.audio_branch.state_dict(),
                    'clap_audio_projection': model.clap_model.audio_projection.state_dict(),
                    'hatr': model.hatr.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc,
                }, os.path.join(fold_dir, "best_model.pth"))
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

        # ── Evaluate on test set ──
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), weights_only=False)
        model.clap_model.audio_branch.load_state_dict(ckpt['clap_audio_branch'])
        model.clap_model.audio_projection.load_state_dict(ckpt['clap_audio_projection'])
        model.hatr.load_state_dict(ckpt['hatr'])

        test_loss, test_acc = validate(model, test_loader, device)
        print(f"[Fold {fold}] Test accuracy: {test_acc:.2f}%")

        fold_result = {
            'fold': fold, 'test_acc': test_acc, 'best_val_acc': best_val_acc,
        }
        all_results.append(fold_result)

        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        mean_acc = np.mean([r['test_acc'] for r in all_results])
        std_acc = np.std([r['test_acc'] for r in all_results])
        print(f"Mean test accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")

if __name__ == "__main__":
    main()
