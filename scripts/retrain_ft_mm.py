"""Retrain ft_mm HATR with CORRECT per-fold fine-tuned CLAP embeddings.

The original ft_mm had data leakage because data_finetune/processed_dataset.csv
pointed all folds to embeddings from a single checkpoint. This script fixes that
by using per-fold embeddings from finetuned_clap_audio_embeddings/fold_X/.

No GPU needed — HATR trains on precomputed 512-dim embeddings.

Usage:
    python retrain_ft_mm.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

from models import BaseClassifier
from dataset_utils import HATRDataset
from losses import CrossEntropyLoss
from evaluate import evaluate_model
from utils import set_seed, build_class_to_topclass_tensor

# Config
NUM_FOLDS = 5
SEED = 42
NUM_EPOCHS = 70
LR = 0.001
BATCH_SIZE = 64
PATIENCE = 10
EARLY_STOP_FACTOR = 3
HIDDEN_SIZE = 128
NUM_CLASSES = 23
NUM_TOP_CLASSES = 5

# Paths
BASE_CSV = "data/processed_dataset.csv"  # baseline CSV with frozen CLAP paths
FT_EMB_BASE = "data/BSD10k-v1.2/features/finetuned_clap_audio_embeddings"
TEXT_EMB_DIR = "data/BSD10k-v1.2/features/clap_text_embeddings"
OUTPUT_DIR = "model_output_finetune_mm_fixed/both"
BASELINE_SPLITS_DIR = "model_output/both"  # use same splits as all other models

CLASS_DICT = "data/class_dict.json"
TOP_CLASS_DICT = "data/top_class_dict.json"


def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load class mappings
    class_dict = json.load(open(CLASS_DICT))
    top_class_dict = json.load(open(TOP_CLASS_DICT))
    class_to_topclass_tensor = build_class_to_topclass_tensor(class_dict, top_class_dict, device)

    # Load baseline CSV
    base_df = pd.read_csv(BASE_CSV)
    print(f"Loaded {len(base_df)} samples from {BASE_CSV}")

    all_results = []

    for fold in range(NUM_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}")
        print(f"{'='*60}")

        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        # Check per-fold embeddings exist
        fold_emb_dir = os.path.join(FT_EMB_BASE, f"fold_{fold}")
        if not os.path.isdir(fold_emb_dir):
            print(f"  ERROR: {fold_emb_dir} not found! Skipping.")
            continue

        # Create fold-specific DataFrame: swap audio_emb_filepath to per-fold
        fold_df = base_df.copy()
        fold_df['audio_emb_filepath'] = fold_df['index'].astype(str).apply(
            lambda sid: os.path.abspath(os.path.join(fold_emb_dir, f"{sid}.npy"))
        )
        # Keep text embeddings pointing to frozen CLAP text
        fold_df['text_emb_filepath'] = fold_df['index'].astype(str).apply(
            lambda sid: os.path.abspath(os.path.join(TEXT_EMB_DIR, f"{sid}.npy"))
        )

        # Verify a sample embedding exists and is correct size
        sample_path = fold_df.iloc[0]['audio_emb_filepath']
        if not os.path.exists(sample_path):
            print(f"  ERROR: Sample embedding missing: {sample_path}")
            continue
        sample_emb = np.load(sample_path)
        print(f"  Embedding shape: {sample_emb.shape} (expected (512,))")
        assert sample_emb.shape == (512,), f"Bad embedding shape: {sample_emb.shape}"

        # Load splits from baseline (same as all other models)
        splits_path = os.path.join(BASELINE_SPLITS_DIR, f"fold_{fold}", "splits.csv")
        splits = pd.read_csv(splits_path)
        train_indices = set(splits[splits['split'] == 'train']['index'].astype(str).tolist())
        val_indices = set(splits[splits['split'] == 'val']['index'].astype(str).tolist())
        test_indices = set(splits[splits['split'] == 'test']['index'].astype(str).tolist())

        fold_df['index'] = fold_df['index'].astype(str)
        train_df = fold_df[fold_df['index'].isin(train_indices)].reset_index(drop=True)
        val_df = fold_df[fold_df['index'].isin(val_indices)].reset_index(drop=True)
        test_df = fold_df[fold_df['index'].isin(test_indices)].sort_values('index').reset_index(drop=True)

        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Save splits for ensemble_evaluate.py compatibility
        all_split_records = []
        for idx in train_df['index']:
            all_split_records.append({'index': idx, 'split': 'train'})
        for idx in val_df['index']:
            all_split_records.append({'index': idx, 'split': 'val'})
        for idx in test_df['index']:
            all_split_records.append({'index': idx, 'split': 'test'})
        pd.DataFrame(all_split_records).to_csv(os.path.join(fold_dir, "splits.csv"), index=False)

        # Create datasets
        train_dataset = HATRDataset(train_df, aug=True)
        val_dataset = HATRDataset(val_df, aug=False)
        test_dataset = HATRDataset(test_df, aug=False)

        _nw = 0 if torch.cuda.is_available() else 4
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=_nw)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=_nw)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=_nw)

        # Create model
        model = BaseClassifier(
            hidden_size=HIDDEN_SIZE, num_classes=NUM_CLASSES,
            emb_size_audio=512, emb_size_text=512,
            dropout=0.2, use_batch_norm=True, mode='both',
            num_residual_blocks=3, use_attention_fusion=True,
            num_top_classes=NUM_TOP_CLASSES
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=PATIENCE
        )
        criterion = CrossEntropyLoss()

        best_val_acc = 0.0
        epochs_no_improve = 0

        for epoch in range(NUM_EPOCHS):
            # Train
            model.train()
            total_loss = 0.0
            total_samples = 0
            for data in train_loader:
                audio_emb = data['audio_embedding'].to(device)
                text_emb = data['text_embedding'].to(device)
                labels = data['class_idx'].to(device)
                bs = labels.size(0)

                optimizer.zero_grad()
                _, class_logits, top_logits, _ = model(audio_emb, text_emb)

                loss = criterion(class_logits, labels)
                if top_logits is not None:
                    top_labels = class_to_topclass_tensor[labels]
                    loss += 0.3 * criterion(top_logits, top_labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bs
                total_samples += bs

            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    audio_emb = data['audio_embedding'].to(device)
                    text_emb = data['text_embedding'].to(device)
                    labels = data['class_idx'].to(device)
                    _, class_logits, _, _ = model(audio_emb, text_emb)
                    preds = class_logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            val_acc = correct / total
            scheduler.step(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_no_improve = 0
                # Save best model
                torch.save({
                    'model_state': model.state_dict(),
                    'config': {
                        'hidden_size': HIDDEN_SIZE, 'num_classes': NUM_CLASSES,
                        'emb_size_audio': 512, 'emb_size_text': 512,
                        'dropout': 0.2, 'use_batch_norm': True, 'mode': 'both',
                        'num_residual_blocks': 3, 'use_attention_fusion': True,
                        'num_top_classes': NUM_TOP_CLASSES
                    },
                    'val_acc': best_val_acc,
                    'epoch': epoch,
                }, os.path.join(fold_dir, "best_model.pth"))
            else:
                epochs_no_improve += 1

            if epoch % 10 == 0 or epochs_no_improve == 0:
                print(f"  Epoch {epoch:3d}: loss={total_loss/total_samples:.4f}, val_acc={100*val_acc:.2f}%, best={100*best_val_acc:.2f}%")

            if epochs_no_improve >= PATIENCE * EARLY_STOP_FACTOR:
                print(f"  Early stopping at epoch {epoch}")
                break

        # Load best model for test evaluation
        ckpt = torch.load(os.path.join(fold_dir, "best_model.pth"), weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()

        # Generate test logits
        logits_list = []
        labels_list = []
        ids_list = []

        with torch.no_grad():
            for data in test_loader:
                audio_emb = data['audio_embedding'].to(device)
                text_emb = data['text_embedding'].to(device)
                _, class_logits, _, _ = model(audio_emb, text_emb)
                logits_list.append(class_logits.cpu().numpy())
                labels_list.append(data['class_idx'].numpy())
                sid = data['sound_id']
                if isinstance(sid, torch.Tensor):
                    ids_list.extend(sid.tolist())
                elif isinstance(sid, (tuple, list)):
                    ids_list.extend(list(sid))
                else:
                    ids_list.append(sid)

        all_logits = np.concatenate(logits_list, axis=0)
        all_labels = np.concatenate(labels_list, axis=0)
        all_ids = np.array(ids_list, dtype=object)

        np.save(os.path.join(fold_dir, "test_logits.npy"), all_logits)
        np.save(os.path.join(fold_dir, "test_labels.npy"), all_labels)
        np.save(os.path.join(fold_dir, "test_ids.npy"), all_ids)

        test_acc = (all_logits.argmax(axis=1) == all_labels).mean()
        print(f"\n  Fold {fold}: test_acc={100*test_acc:.2f}%, best_val_acc={100*best_val_acc:.2f}%")
        all_results.append({'fold': fold, 'test_acc': test_acc, 'val_acc': best_val_acc})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (ft_mm FIXED — per-fold fine-tuned embeddings)")
    print(f"{'='*60}")
    accs = [r['test_acc'] for r in all_results]
    for r in all_results:
        print(f"  Fold {r['fold']}: test={100*r['test_acc']:.2f}%, val={100*r['val_acc']:.2f}%")
    print(f"  Mean: {100*np.mean(accs):.2f}% ± {100*np.std(accs):.2f}%")
    print(f"\nOutput saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
