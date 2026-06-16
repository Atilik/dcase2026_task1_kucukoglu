"""Ensemble evaluation: try all model combinations and find the best."""
import os
import sys
import json
import torch
import numpy as np
import pandas as pd
from itertools import combinations
from torch.utils.data import DataLoader

from models import BaseClassifier
from dataset_utils import HATRDataset
from utils import get_subconfig, build_class_to_topclass_mapping, build_id_to_class_mapping
from evaluate import hierarchical_prf_weighted

# All model directories: (name, model_dir) or (name, model_dir, data_dir)
# Models without data_dir use the default from config.yaml
ALL_MODEL_DIRS = [
    # CLAP models (default data dir)
    ("baseline",   "model_output/both"),
    ("cw_only",    "model_output_cw_only/both"),
    ("hloss_030",  "model_output_clap_hloss_w030/both"),
    ("hloss_070",  "model_output_clap_hloss_w070/both"),
    ("conf3",      "model_output_conf3/both"),
    ("hloss_100",  "model_output_clap_hloss_w100/both"),
    ("hloss_150",  "model_output_clap_hloss_w150/both"),
    ("classw_sq",  "model_output_classw_sqrt/both"),
    ("classw_cw",  "model_output_classw_cw/both"),
    ("cw_hloss07", "model_output_cw_hloss07/both"),
    ("cw_shifted", "model_output_cw_shifted/both"),
    ("cw_binary",  "model_output_cw_binary/both"),
    ("mixup01",    "model_output_mixup01/both"),
    ("mixup02",    "model_output_mixup02/both"),
    ("mixup03",    "model_output_mixup03/both"),
    ("mixup04",    "model_output_mixup04/both"),
    ("m02_cw",     "model_output_mixup02_cw/both"),
    ("m02_hl07",   "model_output_mixup02_hloss07/both"),
    ("m02_clw",    "model_output_mixup02_classw/both"),
    ("bal_max",    "model_output_balanced_max/both"),
    ("bal_med",    "model_output_balanced_med/both"),
    ("xswap",      "model_output_xswap_noise/both"),
    ("combo_aug",  "model_output_combo_aug/both"),
    ("ls01",       "model_output_ls01/both"),
    ("cutmix02",   "model_output_cutmix02/both"),
    # Different encoder models (custom data dirs)
    ("whisper",    "model_output_whisper/both",         "data_whisper"),
    ("convnext",   "model_output_convnext/both",        "data_convnext"),
    ("cnx_mix02",  "model_output_convnext_mix02/both",  "data_convnext"),
    ("matpac",     "model_output_matpac/both",          "data_matpac"),
    # 3-Modality models (CLAP audio + CLAP text + Whisper)
    ("3mod",       "model_output_3mod/both",            "data_3mod"),
    ("3mod_mx02",  "model_output_3mod_mix02/both",      "data_3mod"),
    ("3mod_hl07",  "model_output_3mod_hloss07/both",    "data_3mod"),
    # Contrastive Loss Models
    ("contrastive_02", "model_output_contrastive_02/both"),
    ("contrastive_05", "model_output_contrastive_05/both"),
    # End-to-End Finetuned CLAP
    ("clap_ft",    "model_output_finetune/both"),
    # Strategy A: External Data
    ("ft_mm_fixed", "model_output_finetune_mm_fixed/both"),
    ("ext_weak",   "model_output_ext_weak/both",      "data_ext_weak"),
    ("ext_all",    "model_output_ext_all/both",        "data_ext_all"),
    ("ext_wt",     "model_output_ext_weighted/both",   "data_ext_weighted"),
]

NUM_FOLDS = 5
MIN_COMBO = 2  # minimum models in combo
MAX_COMBO = 7  # maximum models in combo


def load_model(model_dir, fold, device):
    model_path = os.path.join(model_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(model_path):
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = BaseClassifier(**config)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        return model
    except (KeyError, RuntimeError) as e:
        print(f"  WARNING: Cannot load {model_path}: {e} — skipping (needs precomputed logits)")
        return None


def get_test_indices(model_dir, fold):
    splits_path = os.path.join(model_dir, f"fold_{fold}", "splits.csv")
    if not os.path.exists(splits_path):
        return None
    splits = pd.read_csv(splits_path)
    return set(splits[splits['split'] == 'test']['index'].astype(str).tolist())


def verify_splits(valid_models):
    """Verify all models use the same test set per fold."""
    print("\n=== Verifying test splits ===")
    all_match = True
    for fold in range(NUM_FOLDS):
        ref_ids = None
        ref_name = None
        for name, path in valid_models:
            ids = get_test_indices(path, fold)
            if ids is None:
                continue
            if ref_ids is None:
                ref_ids = ids
                ref_name = name
            elif ids != ref_ids:
                print(f"  ⚠️ Fold {fold}: {name} has DIFFERENT test set than {ref_name}!")
                print(f"     {ref_name}: {len(ref_ids)} samples, {name}: {len(ids)} samples")
                print(f"     Diff: {len(ids - ref_ids)} extra, {len(ref_ids - ids)} missing")
                all_match = False
    if all_match:
        print("  ✅ All models use identical test sets per fold")
    return all_match


def collect_all_logits(valid_models, data_dfs, device):
    """Pre-collect logits from all models for all folds."""
    print("\n=== Collecting logits from all models ===")
    # {model_name: {fold: logits_array}}
    all_logits = {}
    fold_labels = {}
    fold_ids = {}
    
    for name, model_dir, data_dir in valid_models:
        full_df = data_dfs[data_dir]
        all_logits[name] = {}
        for fold in range(NUM_FOLDS):
            # Precomputed logits handling for finetuned models
            precomputed_logits_path = os.path.join(model_dir, f"fold_{fold}", "test_logits.npy")
            if os.path.exists(precomputed_logits_path):
                # Load precomputed logits, labels, and ids
                logits = np.load(precomputed_logits_path)
                labels = np.load(os.path.join(model_dir, f"fold_{fold}", "test_labels.npy"))
                ids = np.load(os.path.join(model_dir, f"fold_{fold}", "test_ids.npy"), allow_pickle=True).tolist()
                
                all_logits[name][fold] = logits
                if fold not in fold_labels:
                    fold_labels[fold] = labels
                    fold_ids[fold] = ids
                else:
                    # Verify alignment with existing reference
                    ids_str = [str(x) for x in ids]
                    ref_str = [str(x) for x in fold_ids[fold]]
                    if ids_str != ref_str:
                        print(f"  ⚠️ WARNING: {name} fold {fold} (precomputed) has MISALIGNED samples! Skipping.")
                        del all_logits[name][fold]
                
                # Print accuracy for precomputed logits
                if fold in all_logits.get(name, {}):
                    preds = logits.argmax(axis=1)
                    acc = (preds == fold_labels[fold]).mean()
                    print(f"  {name} fold {fold}: acc={100*acc:.1f}% (precomputed)")
                continue

            model = load_model(model_dir, fold, device)
            if model is None:
                continue
            
            # Get test data
            splits_path = os.path.join(model_dir, f"fold_{fold}", "splits.csv")
            if not os.path.exists(splits_path):
                # Fallback to default splits if missing
                splits_path = f"model_output/both/fold_{fold}/splits.csv"
            splits = pd.read_csv(splits_path)
            test_indices = splits[splits['split'] == 'test']['index'].astype(str).tolist()
            test_df = full_df[full_df['index'].astype(str).isin(test_indices)].sort_values('index').reset_index(drop=True)
            
            test_dataset = HATRDataset(test_df, aug=False)
            _nw = 0 if torch.cuda.is_available() else 4
            test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=_nw)
            
            logits_list = []
            labels_list = []
            ids_list = []
            
            with torch.no_grad():
                for data in test_loader:
                    audio_emb = data['audio_embedding'].to(device)
                    text_emb = data['text_embedding'].to(device)
                    aux_emb = data.get('aux_embedding', None)
                    if aux_emb is not None:
                        aux_emb = aux_emb.to(device)
                    _, class_logits, _, _ = model(audio_emb, text_emb, aux_emb)
                    logits_list.append(class_logits.cpu().numpy())
                    labels_list.append(data['class_idx'].numpy())
                    ids_list.extend(data['sound_id'] if isinstance(data['sound_id'], list) else data['sound_id'].tolist())
            
            all_logits[name][fold] = np.concatenate(logits_list, axis=0)
            fold_labels_local = np.concatenate(labels_list)
            if fold not in fold_labels:
                fold_labels[fold] = fold_labels_local
                fold_ids[fold] = ids_list
            else:
                # Verify sample alignment across models
                ids_str = [str(x) for x in ids_list]
                ref_str = [str(x) for x in fold_ids[fold]]
                if ids_str != ref_str:
                    # Try to reindex: keep only common BSD10k samples in reference order
                    id_to_idx = {str(x): i for i, x in enumerate(ids_list)}
                    common_indices = [id_to_idx[ref_id] for ref_id in ref_str if ref_id in id_to_idx]
                    coverage = len(common_indices) / len(ref_str)
                    if coverage >= 0.99:  # allow up to 1% missing
                        all_logits[name][fold] = all_logits[name][fold][common_indices]
                        print(f"  ℹ️  {name} fold {fold}: reindexed {len(common_indices)}/{len(ids_str)} → {len(ref_str)} ref samples ({100*coverage:.1f}% coverage)")
                    else:
                        print(f"  ⚠️ WARNING: {name} fold {fold} has MISALIGNED samples! Only {100*coverage:.1f}% overlap. Skipping.")
                        del all_logits[name][fold]
                        continue
            
            # Individual accuracy
            preds = all_logits[name][fold].argmax(axis=1)
            acc = (preds == fold_labels[fold]).mean()
            print(f"  {name} fold {fold}: acc={100*acc:.1f}%")
    
    return all_logits, fold_labels, fold_ids


def evaluate_combo(combo_names, all_logits, fold_labels, id_to_class):
    """Evaluate a specific combination of models."""
    fold_hf1s = []
    
    for fold in range(NUM_FOLDS):
        # Check all models have this fold
        logits_stack = []
        for name in combo_names:
            if fold in all_logits[name]:
                logits_stack.append(all_logits[name][fold])
        
        if len(logits_stack) < len(combo_names):
            continue
        
        avg_logits = np.mean(logits_stack, axis=0)
        preds = avg_logits.argmax(axis=1)
        labels = fold_labels[fold]
        
        pred_labels = [id_to_class.get(p, str(p)) for p in preds]
        gt_labels = [id_to_class.get(gt, str(gt)) for gt in labels]
        pred_gt_pairs = list(zip(pred_labels, gt_labels))
        classes = list(set(gt_labels))
        
        h_f1s = []
        for c in classes:
            try:
                _, _, f = hierarchical_prf_weighted(c, pred_gt_pairs, lambda_param=0.75)
                if not np.isnan(f):
                    h_f1s.append(f)
            except:
                continue
        fold_hf1s.append(np.mean(h_f1s))
    
    if not fold_hf1s:
        return 0.0, 0.0
    return np.mean(fold_hf1s), np.std(fold_hf1s)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    default_data_dir = get_subconfig("output_path")
    class_dict = json.load(open(os.path.join(default_data_dir, get_subconfig("class_dict_json"))))
    top_class_dict = json.load(open(os.path.join(default_data_dir, get_subconfig("top_class_dict_json"))))
    id_to_class = build_id_to_class_mapping(class_dict)
    
    # Normalize entries to (name, model_dir, data_dir)
    normalized = []
    for entry in ALL_MODEL_DIRS:
        if len(entry) == 3:
            normalized.append(entry)
        else:
            normalized.append((entry[0], entry[1], default_data_dir))
    
    # Filter to valid dirs
    valid_models = [(n, d, dd) for n, d, dd in normalized if os.path.isdir(d)]
    print(f"Found {len(valid_models)} valid model directories:")
    for n, d, dd in valid_models:
        extra = f" [data: {dd}]" if dd != default_data_dir else ""
        print(f"  {n}: {d}{extra}")
    
    # Pre-load all unique data DFs
    data_dirs = set(dd for _, _, dd in valid_models)
    data_dfs = {}
    for dd in data_dirs:
        csv_path = os.path.join(dd, get_subconfig("processed_dataset_csv"))
        data_dfs[dd] = pd.read_csv(csv_path)
        print(f"  Loaded {csv_path}: {len(data_dfs[dd])} samples")
    
    # Verify splits (pass 2-tuples for compatibility)
    verify_splits([(n, d) for n, d, _ in valid_models])
    
    # Collect all logits
    all_logits, fold_labels, fold_ids = collect_all_logits(valid_models, data_dfs, device)
    
    model_names = [n for n in all_logits if len(all_logits[n]) == NUM_FOLDS]
    print(f"\nModels with all {NUM_FOLDS} folds: {model_names}")
    
    # Individual model scores
    print(f"\n{'='*60}")
    print("INDIVIDUAL MODEL SCORES")
    print(f"{'='*60}")
    individual_scores = {}
    for name in model_names:
        hf1_mean, hf1_std = evaluate_combo([name], all_logits, fold_labels, id_to_class)
        individual_scores[name] = hf1_mean
        print(f"  {name:<15}: {100*hf1_mean:.2f}% ± {100*hf1_std:.2f}%")
    
    # Try all combinations — fast accuracy prescreen, then hF1 for top candidates
    print(f"\n{'='*60}")
    print("ENSEMBLE COMBINATIONS (searching...)")
    print(f"{'='*60}")
    
    # Phase 1: Exhaustive search for small combos (2-5)
    EXHAUSTIVE_MAX = 5
    acc_results = []
    total_combos = 0
    for size in range(MIN_COMBO, min(EXHAUSTIVE_MAX + 1, len(model_names) + 1)):
        size_count = 0
        print(f"  Searching size {size}...", flush=True)
        for combo in combinations(model_names, size):
            total_combos += 1
            size_count += 1
            if size_count % 50000 == 0:
                print(f"    ... {size_count} combos checked (size {size})", flush=True)
            combo_names = list(combo)
            fold_accs = []
            for fold in range(NUM_FOLDS):
                logits_stack = [all_logits[n][fold] for n in combo_names if fold in all_logits[n]]
                if len(logits_stack) < len(combo_names):
                    continue
                avg_logits = np.mean(logits_stack, axis=0)
                preds = avg_logits.argmax(axis=1)
                fold_accs.append((preds == fold_labels[fold]).mean())
            if fold_accs:
                acc_results.append((np.mean(fold_accs), combo))
        print(f"    Done size {size}: {size_count} combos", flush=True)
    
    acc_results.sort(key=lambda x: x[0], reverse=True)
    print(f"  Screened {total_combos} combinations (size 2-{EXHAUSTIVE_MAX})")
    
    # Phase 2: Compute hF1 only for top 50 candidates
    TOP_N = 50
    print(f"  Computing hF1 for top {TOP_N} candidates...")
    results = []
    for _, combo in acc_results[:TOP_N]:
        hf1_mean, hf1_std = evaluate_combo(list(combo), all_logits, fold_labels, id_to_class)
        results.append((hf1_mean, hf1_std, combo))
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 3: Greedy forward selection for larger combos (8-MAX_COMBO)
    print(f"\n  Greedy forward selection (size 8-{MAX_COMBO})...")
    # Start from best size-7 combo
    best_greedy_combo = list(results[0][2]) if results else list(model_names[:2])
    best_greedy_hf1 = results[0][0] if results else 0.0
    
    for target_size in range(EXHAUSTIVE_MAX + 1, MAX_COMBO + 1):
        if len(best_greedy_combo) >= len(model_names):
            break
        best_addition = None
        best_addition_hf1 = best_greedy_hf1
        
        remaining = [m for m in model_names if m not in best_greedy_combo]
        for candidate in remaining:
            test_combo = best_greedy_combo + [candidate]
            hf1_mean, hf1_std = evaluate_combo(test_combo, all_logits, fold_labels, id_to_class)
            if hf1_mean > best_addition_hf1:
                best_addition_hf1 = hf1_mean
                best_addition = candidate
                best_addition_std = hf1_std
        
        if best_addition is not None:
            best_greedy_combo.append(best_addition)
            best_greedy_hf1 = best_addition_hf1
            results.append((best_addition_hf1, best_addition_std, tuple(best_greedy_combo)))
            print(f"    Size {target_size}: +{best_addition} → {100*best_addition_hf1:.2f}%")
        else:
            print(f"    Size {target_size}: no improvement, stopping greedy search")
            break
    
    results.sort(key=lambda x: x[0], reverse=True)
    
    # Top 15 combinations
    print(f"\nTOP 15 ENSEMBLES:")
    print(f"{'Rank':<5} {'hF1':>7} {'±':>6} {'Size':>4}  Models")
    print("-" * 80)
    for i, (hf1, std, combo) in enumerate(results[:15]):
        print(f"  {i+1:<3}  {100*hf1:>6.2f}% {100*std:>5.2f}%  {len(combo):>3}   {', '.join(combo)}")
    
    # Best per size
    print(f"\nBEST PER ENSEMBLE SIZE:")
    for size in range(MIN_COMBO, MAX_COMBO + 1):
        size_results = [r for r in results if len(r[2]) == size]
        if size_results:
            best = max(size_results, key=lambda x: x[0])
            hf1, std, combo = best
            print(f"  Size {size}: {100*hf1:.2f}% ± {100*std:.2f}%  → {', '.join(combo)}")
    
    # Test specific combos requested by user
    SPECIFIC_COMBOS = [
        ["baseline", "cw_only", "hloss_030", "hloss_070", "conf3"],
    ]
    if SPECIFIC_COMBOS:
        print(f"\n{'='*60}")
        print("SPECIFIC COMBO TESTS")
        print(f"{'='*60}")
        for combo in SPECIFIC_COMBOS:
            valid_combo = [m for m in combo if m in model_names]
            if len(valid_combo) == len(combo):
                hf1_mean, hf1_std = evaluate_combo(combo, all_logits, fold_labels, id_to_class)
                print(f"  {', '.join(combo)}")
                print(f"    hF1 = {100*hf1_mean:.2f}% ± {100*hf1_std:.2f}%")
            else:
                missing = set(combo) - set(valid_combo)
                print(f"  {', '.join(combo)}: MISSING {missing}")


if __name__ == "__main__":
    main()
