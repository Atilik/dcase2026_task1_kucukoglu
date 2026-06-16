"""Build external dataset mapping: ESC-50 + FSD50K -> BST classes.

Maps external audio datasets to BST taxonomy, checks for overlap with BSD10k,
and outputs data/external_mapping.csv with the combined mapping.

Run on login node (no GPU needed):
    ./sing <<< "python -u build_external_dataset.py"
"""
import os
import json
import pandas as pd
import numpy as np
from collections import Counter

from external_label_mappings import ESC50_TO_BST, FSD50K_TO_BST

# ── Paths ──
ESC50_META = "/projects/work/marl/datasets/sound_datasets/esc50/ESC-50-master/meta/esc50.csv"
ESC50_AUDIO = "/projects/work/marl/datasets/sound_datasets/esc50/ESC-50-master/audio"

FSD50K_DEV_CSV = "/projects/work/marl/datasets/sound_datasets/fsd50k/FSD50K.ground_truth/dev.csv"
FSD50K_EVAL_CSV = "/projects/work/marl/datasets/sound_datasets/fsd50k/FSD50K.ground_truth/eval.csv"
FSD50K_DEV_AUDIO = "/projects/work/marl/datasets/sound_datasets/fsd50k/FSD50K.dev_audio"
FSD50K_EVAL_AUDIO = "/projects/work/marl/datasets/sound_datasets/fsd50k/FSD50K.eval_audio"

BSD10K_META = "data/BSD10k-v1.2/metadata/BSD10k_metadata.csv"
CLASS_DICT = "data/class_dict.json"
TOP_CLASS_DICT = "data/top_class_dict.json"

OUTPUT_CSV = "data/external_mapping.csv"


def load_class_dicts():
    with open(CLASS_DICT) as f:
        class_dict = json.load(f)
    with open(TOP_CLASS_DICT) as f:
        top_class_dict = json.load(f)
    return class_dict, top_class_dict


def get_top_class(bst_class):
    """Extract top-level class from BST subclass (e.g., 'fx-a' -> 'fx')."""
    return bst_class.split("-")[0]


def process_esc50(class_dict, top_class_dict):
    """Map ESC-50 to BST classes."""
    print("\n" + "=" * 60)
    print("ESC-50 MAPPING")
    print("=" * 60)

    df = pd.read_csv(ESC50_META)
    print(f"Total ESC-50 samples: {len(df)}")

    mapped = []
    unmapped_cats = set()

    for _, row in df.iterrows():
        category = row["category"]
        if category in ESC50_TO_BST:
            bst_class = ESC50_TO_BST[category]
            top_class = get_top_class(bst_class)
            # Extract freesound source ID from filename (e.g., "1-100032-A-0.wav" -> src_file col)
            source_id = f"esc50_{row['filename'].replace('.wav', '')}"
            audio_path = os.path.join(ESC50_AUDIO, row["filename"])

            mapped.append({
                "source_id": source_id,
                "audio_path": audio_path,
                "bst_class": bst_class,
                "class_idx": class_dict[bst_class],
                "top_class": top_class,
                "top_class_idx": top_class_dict[top_class],
                "dataset_source": "ESC-50",
                "freesound_id": str(row["src_file"]),
            })
        else:
            unmapped_cats.add(category)

    print(f"Mapped: {len(mapped)}")
    print(f"Unmapped categories ({len(unmapped_cats)}): {sorted(unmapped_cats)}")

    return mapped


def process_fsd50k(class_dict, top_class_dict):
    """Map FSD50K to BST classes. Only include samples where ALL mapped labels agree."""
    print("\n" + "=" * 60)
    print("FSD50K MAPPING")
    print("=" * 60)

    mapped = []
    conflict_count = 0
    no_mapping_count = 0

    for csv_path, audio_dir, split_name in [
        (FSD50K_DEV_CSV, FSD50K_DEV_AUDIO, "dev"),
        (FSD50K_EVAL_CSV, FSD50K_EVAL_AUDIO, "eval"),
    ]:
        if not os.path.exists(csv_path):
            print(f"Skipping {split_name}: {csv_path} not found")
            continue

        df = pd.read_csv(csv_path)
        print(f"\nFSD50K {split_name}: {len(df)} samples")

        for _, row in df.iterrows():
            fname = str(row["fname"])
            labels = [l.strip() for l in str(row["labels"]).split(",")]

            # Map each label to BST
            bst_classes = set()
            has_mapping = False
            for label in labels:
                if label in FSD50K_TO_BST:
                    bst_classes.add(FSD50K_TO_BST[label])
                    has_mapping = True

            if not has_mapping:
                no_mapping_count += 1
                continue

            # Only keep if ALL mapped labels agree on the same BST class
            if len(bst_classes) > 1:
                conflict_count += 1
                continue

            bst_class = bst_classes.pop()
            top_class = get_top_class(bst_class)
            source_id = f"fsd50k_{fname}"
            audio_path = os.path.join(audio_dir, f"{fname}.wav")

            mapped.append({
                "source_id": source_id,
                "audio_path": audio_path,
                "bst_class": bst_class,
                "class_idx": class_dict[bst_class],
                "top_class": top_class,
                "top_class_idx": top_class_dict[top_class],
                "dataset_source": "FSD50K",
                "freesound_id": fname,
            })

    print(f"\nFSD50K mapped: {len(mapped)}")
    print(f"FSD50K conflicts (multi-BST labels): {conflict_count}")
    print(f"FSD50K no mapping: {no_mapping_count}")

    return mapped


def check_overlap(all_mapped, bsd10k_ids):
    """Remove samples that overlap with BSD10k by Freesound ID."""
    print("\n" + "=" * 60)
    print("OVERLAP CHECK")
    print("=" * 60)

    before = len(all_mapped)
    filtered = [s for s in all_mapped if s["freesound_id"] not in bsd10k_ids]
    overlap = before - len(filtered)

    print(f"BSD10k Freesound IDs: {len(bsd10k_ids)}")
    print(f"External samples before dedup: {before}")
    print(f"Overlapping with BSD10k: {overlap}")
    print(f"External samples after dedup: {len(filtered)}")

    return filtered


def verify_audio_exists(all_mapped):
    """Verify audio files exist and report any missing."""
    print("\n" + "=" * 60)
    print("AUDIO FILE VERIFICATION")
    print("=" * 60)

    missing = 0
    valid = []
    for s in all_mapped:
        if os.path.exists(s["audio_path"]):
            valid.append(s)
        else:
            missing += 1

    print(f"Valid audio files: {len(valid)}")
    print(f"Missing audio files: {missing}")
    return valid


def print_distribution(all_mapped, class_dict):
    """Print BST class distribution, highlighting weak classes."""
    print("\n" + "=" * 60)
    print("BST CLASS DISTRIBUTION")
    print("=" * 60)

    class_counts = Counter(s["bst_class"] for s in all_mapped)
    # Sort by BST class name
    idx_to_class = {v: k for k, v in class_dict.items()}

    weak_classes = {"sp-c", "ss-i"}  # fx-ex removed: no external data maps to "experimental"

    print(f"\n{'Class':<8} {'Count':>6}  {'Source Distribution'}")
    print("-" * 60)

    for bst_class in sorted(class_counts.keys()):
        count = class_counts[bst_class]
        # Source breakdown
        sources = Counter(s["dataset_source"] for s in all_mapped if s["bst_class"] == bst_class)
        source_str = ", ".join(f"{k}: {v}" for k, v in sorted(sources.items()))
        marker = " ⭐ WEAK" if bst_class in weak_classes else ""
        print(f"{bst_class:<8} {count:>6}  {source_str}{marker}")

    # Top-level distribution
    print(f"\n{'Top Class':<12} {'Count':>6}")
    print("-" * 30)
    top_counts = Counter(s["top_class"] for s in all_mapped)
    for tc in sorted(top_counts.keys()):
        print(f"{tc:<12} {top_counts[tc]:>6}")

    print(f"\nTotal external samples: {len(all_mapped)}")

    # Highlight weak classes
    print(f"\n{'=' * 40}")
    print("WEAK CLASS SUMMARY:")
    for wc in sorted(weak_classes):
        count = class_counts.get(wc, 0)
        print(f"  {wc}: {count} samples")


def main():
    class_dict, top_class_dict = load_class_dicts()
    print(f"BST classes: {len(class_dict)}")
    print(f"Top classes: {len(top_class_dict)}")

    # Load BSD10k IDs for overlap check
    bsd10k_df = pd.read_csv(BSD10K_META)
    # BSD10k uses UUIDs as index, but the original Freesound ID might be in
    # the 'freesound_id' column or might just be the 'id' column.
    # Check what columns exist
    print(f"\nBSD10k columns: {list(bsd10k_df.columns)}")
    print(f"BSD10k shape: {bsd10k_df.shape}")
    print(f"BSD10k first row:\n{bsd10k_df.iloc[0]}")

    # Try to extract Freesound IDs from BSD10k
    # The 'sound_id' column in BSD10k contains Freesound sound IDs
    if "sound_id" in bsd10k_df.columns:
        bsd10k_ids = set(bsd10k_df["sound_id"].astype(str).tolist())
    elif "id" in bsd10k_df.columns:
        bsd10k_ids = set(bsd10k_df["id"].astype(str).tolist())
    elif "freesound_id" in bsd10k_df.columns:
        bsd10k_ids = set(bsd10k_df["freesound_id"].astype(str).tolist())
    else:
        print("WARNING: Cannot determine BSD10k ID column for overlap check!")
        bsd10k_ids = set()

    print(f"BSD10k IDs (first 5): {list(bsd10k_ids)[:5]}")

    # Process datasets
    esc50_mapped = process_esc50(class_dict, top_class_dict)
    fsd50k_mapped = process_fsd50k(class_dict, top_class_dict)

    all_mapped = esc50_mapped + fsd50k_mapped
    print(f"\nTotal before overlap removal: {len(all_mapped)}")

    # Remove BSD10k overlaps
    all_mapped = check_overlap(all_mapped, bsd10k_ids)

    # Verify audio files exist
    all_mapped = verify_audio_exists(all_mapped)

    # Print distribution
    print_distribution(all_mapped, class_dict)

    # Save
    df = pd.DataFrame(all_mapped)
    df = df.drop(columns=["freesound_id"])  # don't need this in final CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved to {OUTPUT_CSV}")
    print(f"Shape: {df.shape}")


if __name__ == "__main__":
    main()
