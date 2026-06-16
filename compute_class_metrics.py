"""Compute per-class hP, hR, hF using raw logits (same as ensemble_evaluate.py).

Uses forward pass for HATR models + precomputed test_logits.npy for clap_ft.

Usage: ./sing <<< "python -u compute_class_metrics.py"
"""
import os, json, re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from models import BaseClassifier
from dataset_utils import HATRDataset
from utils import get_subconfig, build_id_to_class_mapping
from evaluate import hierarchical_prf_weighted

NUM_FOLDS = 5

SUBMISSIONS = {
    1: {
        "name": "5-model ensemble (mixup03+m02_hl07+bal_med+cnx_mix02+clap_ft)",
        "models": [
            ("mixup03",   "model_output_mixup03/both",          "data"),
            ("m02_hl07",  "model_output_mixup02_hloss07/both",  "data"),
            ("bal_med",   "model_output_balanced_med/both",      "data"),
            ("cnx_mix02", "model_output_convnext_mix02/both",   "data_convnext"),
            ("clap_ft",   "model_output_finetune/both",         "data"),
        ],
        "yaml": "Kucukoglu_NYU_task1_1.meta.yaml",
    },
    2: {
        "name": "5-model CLAP+ConvNeXt",
        "models": [
            ("hloss_070", "model_output_clap_hloss_w070/both",  "data"),
            ("m02_clw",   "model_output_mixup02_classw/both",   "data"),
            ("bal_med",   "model_output_balanced_med/both",      "data"),
            ("combo_aug", "model_output_combo_aug/both",         "data"),
            ("cnx_mix02", "model_output_convnext_mix02/both",   "data_convnext"),
        ],
        "yaml": "Kucukoglu_NYU_task1_2.meta.yaml",
    },
    3: {
        "name": "Single model: xswap",
        "models": [
            ("xswap", "model_output_xswap_noise/both", "data"),
        ],
        "yaml": "Kucukoglu_NYU_task1_3.meta.yaml",
    },
    4: {
        "name": "Single model: 3mod_mx02",
        "models": [
            ("3mod_mx02", "model_output_3mod_mix02/both", "data_3mod"),
        ],
        "yaml": "Kucukoglu_NYU_task1_4.meta.yaml",
    },
}


def collect_fold_logits(model_name, model_dir, data_dir, fold, device):
    """Collect raw logits for a fold — same logic as ensemble_evaluate.py."""
    # Load precomputed logits if available
    precomputed = os.path.join(model_dir, f"fold_{fold}", "test_logits.npy")
    if os.path.exists(precomputed):
        logits = np.load(precomputed)
        labels = np.load(os.path.join(model_dir, f"fold_{fold}", "test_labels.npy"))
        return logits, labels

    # Load model
    model_path = os.path.join(model_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(model_path):
        return None, None

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = BaseClassifier(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    # Get test split
    splits_path = os.path.join(model_dir, f"fold_{fold}", "splits.csv")
    if not os.path.exists(splits_path):
        return None, None

    splits = pd.read_csv(splits_path)
    test_indices = splits[splits['split'] == 'test']['index'].astype(str).tolist()

    full_df = pd.read_csv(os.path.join(data_dir, "processed_dataset.csv"))
    test_df = full_df[full_df['index'].astype(str).isin(test_indices)].sort_values('index').reset_index(drop=True)

    dataset = HATRDataset(test_df, aug=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    logits_list = []
    labels_list = []
    with torch.no_grad():
        for data in loader:
            audio_emb = data['audio_embedding'].to(device)
            text_emb = data['text_embedding'].to(device)
            aux_emb = data.get('aux_embedding', None)
            if aux_emb is not None:
                aux_emb = aux_emb.to(device)
            _, class_logits, _, _ = model(audio_emb, text_emb, aux_emb)
            logits_list.append(class_logits.cpu().numpy())
            labels_list.append(data['class_idx'].numpy())

    del model
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list)


def compute_submission_metrics(sub_config, id_to_class, all_classes, device):
    """Compute per-class hP, hR, hF for a submission using raw logits."""
    fold_metrics = []

    for fold in range(NUM_FOLDS):
        logits_stack = []
        gt_labels = None

        for model_name, model_dir, data_dir in sub_config["models"]:
            logits, labels = collect_fold_logits(model_name, model_dir, data_dir, fold, device)
            if logits is None:
                print(f"    {model_name} fold {fold}: MISSING")
                continue

            logits_stack.append(logits)
            if gt_labels is None:
                gt_labels = labels
            print(f"    {model_name} fold {fold}: ✅ ({logits.shape[0]} samples)")

        if not logits_stack or gt_labels is None:
            continue

        # Align lengths
        min_len = min(l.shape[0] for l in logits_stack)
        logits_stack = [l[:min_len] for l in logits_stack]
        gt_labels = gt_labels[:min_len]

        avg_logits = np.mean(logits_stack, axis=0)
        preds = avg_logits.argmax(axis=1)

        pred_labels = [id_to_class[int(p)] for p in preds]
        gt_label_strs = [id_to_class[int(g)] for g in gt_labels]
        pred_gt_pairs = list(zip(pred_labels, gt_label_strs))

        # Per-class metrics
        class_metrics = {}
        for c in all_classes:
            try:
                p, r, f = hierarchical_prf_weighted(c, pred_gt_pairs, lambda_param=0.75)
                if not np.isnan(f):
                    class_metrics[c] = (p, r, f)
            except Exception:
                pass

        if class_metrics:
            hP = np.mean([m[0] for m in class_metrics.values()])
            hR = np.mean([m[1] for m in class_metrics.values()])
            hF = np.mean([m[2] for m in class_metrics.values()])
            fold_metrics.append({"overall": (hP, hR, hF), "class_wise": class_metrics})
            print(f"    Fold {fold}: hP={100*hP:.2f} hR={100*hR:.2f} hF={100*hF:.2f}")

    if not fold_metrics:
        return None

    # Average across folds
    result = {"overall": {}, "class_wise": {}}
    result["overall"] = {
        "hP": round(float(np.mean([m["overall"][0] for m in fold_metrics])), 4),
        "hR": round(float(np.mean([m["overall"][1] for m in fold_metrics])), 4),
        "hF": round(float(np.mean([m["overall"][2] for m in fold_metrics])), 4),
    }
    for c in all_classes:
        vals = [m["class_wise"][c] for m in fold_metrics if c in m["class_wise"]]
        if vals:
            result["class_wise"][c] = {
                "hP": round(float(np.mean([v[0] for v in vals])), 4),
                "hR": round(float(np.mean([v[1] for v in vals])), 4),
                "hF": round(float(np.mean([v[2] for v in vals])), 4),
            }
    return result


def update_yaml(yaml_path, metrics):
    """Update meta YAML file with computed metrics."""
    with open(yaml_path, 'r') as f:
        content = f.read()

    # Update overall
    content = re.sub(
        r'(overall:\s*\n\s*hP:)\s*(?:!!null|[\d.]+)',
        f'\\1 {metrics["overall"]["hP"]}',
        content, count=1
    )
    content = re.sub(
        r'(overall:\s*\n\s*hP:.*\n\s*hR:)\s*(?:!!null|[\d.]+)',
        f'\\1 {metrics["overall"]["hR"]}',
        content, count=1
    )
    content = re.sub(
        r'(overall:\s*\n\s*hP:.*\n\s*hR:.*\n\s*hF:)\s*(?:!!null|[\d.]+)',
        f'\\1 {metrics["overall"]["hF"]}',
        content, count=1
    )

    # Update per-class
    for cls, m in metrics["class_wise"].items():
        pattern = f'({re.escape(cls)}:\\s*\\n\\s*hP:)\\s*(?:!!null|[\\d.]+)(\\s*\\n\\s*hR:)\\s*(?:!!null|[\\d.]+)(\\s*\\n\\s*hF:)\\s*(?:!!null|[\\d.]+)'
        replacement = f'\\1 {m["hP"]}\\2 {m["hR"]}\\3 {m["hF"]}'
        content = re.sub(pattern, replacement, content)

    with open(yaml_path, 'w') as f:
        f.write(content)
    print(f"  ✅ Updated: {yaml_path}")


def main():
    device = torch.device("cpu")  # CPU is fine for inference on embeddings
    class_dict = json.load(open("data/class_dict.json"))
    id_to_class = {v: k for k, v in class_dict.items()}
    all_classes = sorted(class_dict.keys())

    for sub_id, sub in SUBMISSIONS.items():
        print(f"\n{'='*60}")
        print(f"Submission {sub_id}: {sub['name']}")
        print(f"{'='*60}")

        metrics = compute_submission_metrics(sub, id_to_class, all_classes, device)
        if metrics:
            print(f"\n  Overall: hP={100*metrics['overall']['hP']:.2f}% hR={100*metrics['overall']['hR']:.2f}% hF={100*metrics['overall']['hF']:.2f}%")
            print(f"  Per-class ({len(metrics['class_wise'])} classes):")
            for cls in sorted(metrics['class_wise'].keys()):
                m = metrics['class_wise'][cls]
                print(f"    {cls:6s}: hP={m['hP']:.4f}  hR={m['hR']:.4f}  hF={m['hF']:.4f}")
            update_yaml(sub["yaml"], metrics)
        else:
            print(f"  ERROR: Could not compute metrics!")


if __name__ == "__main__":
    main()
