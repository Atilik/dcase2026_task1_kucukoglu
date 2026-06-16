"""Check class distribution in the test folds and optionally create a balanced test2."""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from collections import Counter

SEED = 1821
K_FOLDS = 5

# Load the standard BSD10k processed dataset
df = pd.read_csv("data/processed_dataset.csv")
bsd10k = df[df['dataset_source'] == 'BSD10k-v1.2'].reset_index(drop=True)

with open("data/class_dict.json") as f:
    class_dict = json.load(f)
idx_to_class = {v: k for k, v in class_dict.items()}

print(f"BSD10k total: {len(bsd10k)} samples, {len(class_dict)} classes")

labels = bsd10k['class_idx'].tolist()
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

# Check test set distribution for each fold
for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
    test_labels = [labels[i] for i in test_idx]
    test_counts = Counter(test_labels)
    
    print(f"\n{'='*60}")
    print(f"FOLD {fold} — Test set: {len(test_idx)} samples")
    print(f"{'='*60}")
    print(f"{'Class':<8} {'Count':>6} {'Pct':>6}")
    print("-"*24)
    
    min_count = float('inf')
    max_count = 0
    for cls_name in sorted(class_dict.keys()):
        cls_idx = class_dict[cls_name]
        count = test_counts.get(cls_idx, 0)
        pct = 100.0 * count / len(test_idx)
        min_count = min(min_count, count)
        max_count = max(max_count, count)
        marker = " ⭐" if cls_name in ('sp-c', 'ss-i', 'fx-ex') else ""
        print(f"{cls_name:<8} {count:>6} {pct:>5.1f}%{marker}")
    
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    print(f"\nMin: {min_count}, Max: {max_count}, Ratio: {imbalance_ratio:.1f}x")

# Now show what a balanced test set would look like
print(f"\n{'='*60}")
print("BALANCED TEST SET ANALYSIS")
print(f"{'='*60}")

# For a balanced test: subsample each class to min_count per fold
# OR: compute metrics on the full test but weight by inverse class frequency
# The DCASE eval set is balanced = equal samples per class

# Find the minimum class count across all folds' test sets
fold0_trainval, fold0_test = list(skf.split(np.zeros(len(labels)), labels))[0]
test_labels_fold0 = [labels[i] for i in fold0_test]
test_counts_fold0 = Counter(test_labels_fold0)
min_per_class = min(test_counts_fold0.values())
print(f"\nFold 0 min samples per class: {min_per_class}")
print(f"Balanced test2 would have: {min_per_class * len(class_dict)} samples ({min_per_class}/class)")
print(f"vs current test: {len(fold0_test)} samples")
