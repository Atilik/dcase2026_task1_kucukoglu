"""Fine-tune CLAP audio encoder end-to-end for DCASE 2026 Task 1.

Uses LAION-CLAP with the audio encoder trainable (discriminative LR).
Text embeddings stay frozen (precomputed .npy).

Usage:
    # First install: ./singrw <<< "pip install laion-clap librosa soundfile"
    # Cache model on login node: ./sing <<< "python -c 'import laion_clap; m=laion_clap.CLAP_Module(enable_fusion=True,amodel=\"HTSAT-tiny\"); m.load_ckpt()'"
    # Run on GPU: ./sing <<< "python -u finetune_clap.py"
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
import torchaudio

from models import BaseClassifier
from losses import CrossEntropyLoss
from utils import set_seed

# ── Config ──
SEED = 1821  # Same as baseline for identical splits
K_FOLDS = 5
NUM_EPOCHS = 30
PATIENCE = 5
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2  # effective batch = 16*2 = 32
CLAP_LR = 1e-5
HATR_LR = 1e-3
WEIGHT_DECAY = 1e-4
CLAP_SR = 48000  # CLAP expects 48kHz
MAX_DURATION = 10  # seconds
MAX_SAMPLES = CLAP_SR * MAX_DURATION  # 480000
MIXUP_ALPHA = 0.2  # embedding-level mixup (same as frozen baseline)

# Data paths
METADATA_CSV = "data/processed_dataset.csv"
AUDIO_DIR = "data/BSD10k-v1.2/audio"
TEXT_EMB_DIR = "data/BSD10k-v1.2/features/clap_text_embeddings"
CLASS_DICT_PATH = "data/class_dict.json"
TOP_CLASS_DICT_PATH = "data/top_class_dict.json"
OUTPUT_DIR = "model_output_finetune"

# Which fold to run (set to None for all folds, or 0 for fold-0 only test)
FOLD_ONLY = None  # Change to 0 for testing


# ── CLAP Audio Preprocessor ──
class CLAPAudioPreprocessor:
    """Replicates CLAP's audio preprocessing (mel spectrogram computation).
    
    This is NOT differentiable — it's just feature extraction.
    The gradients flow through audio_branch and audio_projection, not here.
    """
    def __init__(self, sample_rate=48000, window_size=1024, hop_size=480,
                 mel_bins=64, fmin=50, fmax=14000, enable_fusion=True):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.mel_bins = mel_bins
        self.fmin = fmin
        self.fmax = fmax
        self.enable_fusion = enable_fusion
        self.target_length = 1012  # CLAP's default target length for 10s at 48kHz
        
        # torchaudio mel spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            hop_length=hop_size,
            n_mels=mel_bins,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
    
    def __call__(self, waveform):
        """Convert waveform tensor to log mel spectrogram.
        
        Args:
            waveform: (num_samples,) float32 tensor at 48kHz
        Returns:
            fbank: (target_length, mel_bins) log mel spectrogram
        """
        # Compute mel spectrogram
        mel = self.mel_transform(waveform)  # (mel_bins, time)
        
        # Convert to log scale
        fbank = torch.log(mel + 1e-7)  # (mel_bins, time)
        fbank = fbank.T  # (time, mel_bins)
        
        # Pad or truncate to target length
        n_frames = fbank.shape[0]
        if n_frames < self.target_length:
            pad = torch.zeros(self.target_length - n_frames, self.mel_bins)
            fbank = torch.cat([fbank, pad], dim=0)
        elif n_frames > self.target_length:
            fbank = fbank[:self.target_length, :]
        
        return fbank  # (target_length, mel_bins)


# ── Dataset ──
class FinetuneCLAPDataset(Dataset):
    """Dataset that loads raw WAV audio + precomputed text .npy embeddings."""

    def __init__(self, dataframe, audio_dir, text_emb_dir, preprocessor,
                 target_sr=48000, max_duration=10, training=True):
        self.dataframe = dataframe
        self.audio_dir = audio_dir
        self.text_emb_dir = text_emb_dir
        self.preprocessor = preprocessor
        self.target_sr = target_sr
        self.max_samples = target_sr * max_duration
        self.training = training

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        sound_id = sample['index']

        # Load raw audio
        audio_path = os.path.join(self.audio_dir, f"{sound_id}.wav")
        try:
            waveform, sr = sf.read(audio_path, dtype='float32')
        except Exception:
            audio_path = os.path.join(self.audio_dir, f"{sound_id}.flac")
            waveform, sr = sf.read(audio_path, dtype='float32')

        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # Resample to 48kHz if needed
        if sr != self.target_sr:
            import torchaudio.functional as F
            waveform = F.resample(torch.tensor(waveform), orig_freq=sr, new_freq=self.target_sr).numpy()

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

        # Build mel_fusion: (4, time, mel_bins) — CLAP fusion format
        # Channel 0 = real mel, channels 1-3 = zero (since clips are not "longer")
        mel_fusion = fbank.unsqueeze(0).repeat(4, 1, 1)  # (4, time, mel_bins)
        mel_fusion[1:, :, :] = 0.0  # zero channels 1-3 for non-longer clips

        # Load text embedding
        text_path = os.path.join(self.text_emb_dir, f"{sound_id}.npy")
        text_emb = torch.tensor(np.load(text_path), dtype=torch.float32)

        return {
            'mel_fusion': mel_fusion,
            'waveform': waveform,  # keep for fusion mode
            'text_embedding': text_emb,
            'class_idx': sample['class_idx'],
            'top_class_idx': sample['top_class_idx'],
            'sound_id': sound_id,
        }


# ── Model ──
class CLAPFinetuneModel(nn.Module):
    """Wraps CLAP audio encoder + HATR classifier for end-to-end fine-tuning."""

    def __init__(self, clap_module, hatr_model, freeze_early_layers=True):
        super().__init__()
        # Extract the internal CLAP model
        self.clap_model = clap_module.model
        self.hatr = hatr_model
        self.enable_fusion = clap_module.enable_fusion
        
        # Count and optionally freeze early layers
        total_params = 0
        frozen_params = 0
        for name, param in self.clap_model.audio_branch.named_parameters():
            total_params += 1
            if freeze_early_layers:
                # Freeze patch embedding and first 2 HTSAT layers
                if any(k in name for k in ['patch_embed', 'norm_pre', 
                                            'layers.0.', 'layers.1.']):
                    param.requires_grad = False
                    frozen_params += 1
        
        # Also freeze text branch entirely
        for param in self.clap_model.text_branch.parameters():
            param.requires_grad = False
        for param in self.clap_model.text_projection.parameters():
            param.requires_grad = False
            
        print(f"  Audio branch: {frozen_params}/{total_params} params frozen")
        trainable_audio = sum(p.numel() for p in self.clap_model.audio_branch.parameters() if p.requires_grad)
        trainable_proj = sum(p.numel() for p in self.clap_model.audio_projection.parameters() if p.requires_grad)
        trainable_hatr = sum(p.numel() for p in self.hatr.parameters())
        print(f"  Trainable: audio_branch={trainable_audio:,}, audio_proj={trainable_proj:,}, hatr={trainable_hatr:,}")

    def forward(self, mel_fusion, waveform, text_emb):
        """
        Args:
            mel_fusion: (batch, 4, time, mel_bins) 4-channel mel for fusion model
            waveform: (batch, num_samples) raw audio (for fusion)
            text_emb: (batch, 512) precomputed text embedding
        Returns:
            z, class_logit, top_class_logit, attn_scores
        """
        # Build input dict matching what CLAP's audio_branch expects
        # The fusion HTSAT needs: mel_fusion (4-ch mel), waveform, longer (bool)
        input_dict = {
            'mel_fusion': mel_fusion,
            'waveform': waveform,
            'longer': torch.tensor([False] * mel_fusion.shape[0]).to(mel_fusion.device),
        }
        
        # Forward through CLAP audio branch (HTSAT encoder)
        audio_output = self.clap_model.audio_branch(input_dict, mixup_lambda=None)
        
        # Project to 512-dim embedding space
        audio_emb = self.clap_model.audio_projection(audio_output["embedding"])
        
        # Forward through HATR classifier
        z, class_logit, top_class_logit, attn_scores = self.hatr(audio_emb, text_emb)
        return z, class_logit, top_class_logit, attn_scores, audio_emb


def init_hatr_weights(model):
    """Initialize only HATR weights (CLAP weights are pretrained)."""
    for name, param in model.hatr.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


def train_one_epoch(model, train_loader, optimizer, criterion, device,
                    accumulation_steps=1, epoch=0, num_epochs=30,
                    mixup_alpha=0.0):
    """Train for one epoch with gradient accumulation and optional mixup."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    optimizer.zero_grad()

    for step, data in enumerate(train_loader):
        mel_fusion = data['mel_fusion'].to(device)
        waveform = data['waveform'].to(device)
        text_emb = data['text_embedding'].to(device)
        labels = data['class_idx'].to(device)
        batch_size = labels.size(0)

        # Get CLAP audio embedding + HATR output
        z, class_logit, _, _, audio_emb = model(mel_fusion, waveform, text_emb)

        # Embedding-level mixup (after CLAP, before loss)
        if mixup_alpha > 0:
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            index = torch.randperm(batch_size).to(device)
            mixed_audio_emb = lam * audio_emb + (1 - lam) * audio_emb[index]
            mixed_text_emb = lam * text_emb + (1 - lam) * text_emb[index]
            # Re-run through HATR only (CLAP already ran)
            _, class_logit, _, _ = model.hatr(mixed_audio_emb, mixed_text_emb)
            loss_a = criterion(class_logit, labels)
            loss_b = criterion(class_logit, labels[index])
            loss = (lam * loss_a + (1 - lam) * loss_b) / accumulation_steps
        else:
            loss = criterion(class_logit, labels) / accumulation_steps

        loss.backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps * batch_size
        _, predicted = class_logit.max(1)
        total_correct += predicted.eq(labels).sum().item()
        total_samples += batch_size

        if (step + 1) % 20 == 0:
            print(f"    [{epoch+1}/{num_epochs}] Step {step+1}/{len(train_loader)}, "
                  f"Loss: {total_loss/total_samples:.4f}, "
                  f"Acc: {100.*total_correct/total_samples:.1f}%", flush=True)

    return total_loss / total_samples, 100. * total_correct / total_samples


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for data in val_loader:
        mel_fusion = data['mel_fusion'].to(device)
        waveform = data['waveform'].to(device)
        text_emb = data['text_embedding'].to(device)
        labels = data['class_idx'].to(device)
        batch_size = labels.size(0)

        z, class_logit, _, _, _ = model(mel_fusion, waveform, text_emb)
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

    if not torch.cuda.is_available():
        print("ERROR: No GPU detected! CLAP fine-tuning requires GPU.")
        print("Use: srun --gres=gpu:1 ...")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load metadata ──
    full_df = pd.read_csv(METADATA_CSV)
    print(f"Dataset: {len(full_df)} samples")

    with open(CLASS_DICT_PATH) as f:
        class_dict = json.load(f)
    with open(TOP_CLASS_DICT_PATH) as f:
        top_class_dict = json.load(f)
    num_classes = len(class_dict)
    print(f"Classes: {num_classes}")

    # ── Load CLAP model ──
    print("Loading CLAP model...")
    import laion_clap
    from huggingface_hub import hf_hub_download
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_module.load_ckpt(clap_ckpt)
    print(f"CLAP model loaded from {clap_ckpt} (fusion=True, matching BSD10k baseline)")

    # Print structure for debugging
    print("\nCLAP model structure:")
    for name, child in clap_module.model.named_children():
        nparams = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({nparams:,} params)")

    # ── Audio preprocessor ──
    preprocessor = CLAPAudioPreprocessor(
        sample_rate=CLAP_SR, enable_fusion=clap_module.enable_fusion
    )

    # ── K-fold CV ──
    labels = full_df["class_idx"].tolist()
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

    all_results = []
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        if FOLD_ONLY is not None and fold != FOLD_ONLY:
            continue
            
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        # Split train/val (same as baseline)
        trainval_labels = [labels[i] for i in trainval_idx]
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
        train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
        train_idx = [trainval_idx[i] for i in train_idx_rel]
        val_idx = [trainval_idx[i] for i in val_idx_rel]

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        test_df = full_df.iloc[test_idx].reset_index(drop=True)
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # ── Datasets & Loaders ──
        train_dataset = FinetuneCLAPDataset(
            train_df, AUDIO_DIR, TEXT_EMB_DIR, preprocessor,
            target_sr=CLAP_SR, max_duration=MAX_DURATION, training=True
        )
        val_dataset = FinetuneCLAPDataset(
            val_df, AUDIO_DIR, TEXT_EMB_DIR, preprocessor,
            target_sr=CLAP_SR, max_duration=MAX_DURATION, training=False
        )
        test_dataset = FinetuneCLAPDataset(
            test_df, AUDIO_DIR, TEXT_EMB_DIR, preprocessor,
            target_sr=CLAP_SR, max_duration=MAX_DURATION, training=False
        )

        # Use multiple workers for data loading to keep GPU busy
        _nw = 8
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   drop_last=True, num_workers=_nw, pin_memory=True,
                                   persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=_nw, pin_memory=True,
                                 persistent_workers=True, prefetch_factor=4)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                  num_workers=_nw, pin_memory=True,
                                  persistent_workers=True, prefetch_factor=4)

        # ── Build model ──
        # Reload CLAP for each fold (fresh weights)
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

        # ── Optimizer with discriminative LR ──
        audio_branch_params = [p for p in model.clap_model.audio_branch.parameters() if p.requires_grad]
        audio_proj_params = list(model.clap_model.audio_projection.parameters())
        hatr_params = list(model.hatr.parameters())

        optimizer = torch.optim.AdamW([
            {'params': audio_branch_params, 'lr': CLAP_LR, 'weight_decay': WEIGHT_DECAY},
            {'params': audio_proj_params, 'lr': CLAP_LR * 5, 'weight_decay': WEIGHT_DECAY},
            {'params': hatr_params, 'lr': HATR_LR, 'weight_decay': 1e-5},
        ])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
        criterion = CrossEntropyLoss(label_smoothing=0.01)

        # ── Training loop ──
        fold_dir = os.path.join(OUTPUT_DIR, "both", f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        best_val_acc = 0.0
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                accumulation_steps=ACCUMULATION_STEPS,
                epoch=epoch, num_epochs=NUM_EPOCHS,
                mixup_alpha=MIXUP_ALPHA
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            lr_audio = optimizer.param_groups[0]['lr']
            lr_hatr = optimizer.param_groups[2]['lr']
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.2f}% | "
                  f"LR: audio={lr_audio:.2e} hatr={lr_hatr:.2e}", flush=True)

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
                print(f"  ✓ New best model saved (val_acc={val_acc:.2f}%)", flush=True)
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

        test_loss, test_acc = validate(model, test_loader, criterion, device)
        print(f"[Fold {fold}] Test accuracy: {test_acc:.2f}%")

        # Collect predictions for detailed metrics
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for data in test_loader:
                mel_fusion = data['mel_fusion'].to(device)
                waveform = data['waveform'].to(device)
                text_emb = data['text_embedding'].to(device)
                labels_batch = data['class_idx'].to(device)
                _, class_logit, _, _, _ = model(mel_fusion, waveform, text_emb)
                _, predicted = class_logit.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels_batch.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        np.save(os.path.join(fold_dir, "test_predictions.npy"), all_preds)
        np.save(os.path.join(fold_dir, "test_labels.npy"), all_labels)

        fold_result = {
            'fold': fold, 'test_acc': test_acc, 'best_val_acc': best_val_acc,
        }
        all_results.append(fold_result)
        print(f"Fold {fold}: {fold_result}")

        # Free GPU memory
        del model, optimizer, scheduler
        torch.cuda.empty_cache()

    # ── Summary ──
    if all_results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        mean_acc = np.mean([r['test_acc'] for r in all_results])
        std_acc = np.std([r['test_acc'] for r in all_results])
        print(f"Mean test accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        for r in all_results:
            print(f"  Fold {r['fold']}: test={r['test_acc']:.2f}%, val_best={r['best_val_acc']:.2f}%")

        with open(os.path.join(OUTPUT_DIR, "summary_metrics.txt"), 'w') as f:
            f.write(f"CLAP Fine-tuning Results\n")
            f.write(f"Mean accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%\n\n")
            for r in all_results:
                f.write(f"Fold {r['fold']}: test={r['test_acc']:.2f}%, val_best={r['best_val_acc']:.2f}%\n")


if __name__ == "__main__":
    main()
