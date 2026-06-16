import os
import pandas as pd
import json
from utils import get_subconfig

"""
The dataset is built from existing embedding (.npy) files identified by sound_id.
A sample is included only if both audio and text embeddings exist and a matching metadata 
entry is found. The metadata is used to assign class and top-class labels, which are also saved as JSON files.
By default, we exclude top-level classes and classes that belong to the "-other" category.
"""

# --- Filepaths ---
dataset_name = get_subconfig("active_dataset")
datasets_config = get_subconfig("datasets")
metadata_csv = datasets_config[dataset_name]["metadata_csv"]
audio_emb_folder = datasets_config[dataset_name]["audio_emb_folder"]
text_emb_folder = datasets_config[dataset_name]["text_emb_folder"]
aux_emb_folder = datasets_config[dataset_name].get("aux_emb_folder", None)

output_path = get_subconfig("output_path")
os.makedirs(output_path, exist_ok=True)
processed_dataset_csv = os.path.join(output_path, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(output_path, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(output_path, get_subconfig("top_class_dict_json"))
top_class_subclass_dict_json = os.path.join(output_path, get_subconfig("top_class_subclass_dict_json"))

# --- Load metadata ---
df = pd.read_csv(metadata_csv)
df['sound_id'] = df['sound_id'].astype(str).str.strip()
df['_audio_emb_folder'] = os.path.abspath(audio_emb_folder)
df['_text_emb_folder'] = os.path.abspath(text_emb_folder)
df['_aux_emb_folder'] = os.path.abspath(aux_emb_folder) if aux_emb_folder else ''
df['_source'] = dataset_name

print(f"Examining original data from {dataset_name}:")
print(f"  Total rows: {len(df)}")
print(f"  Unique classes: {df['class'].nunique()}")

# --- Load extra datasets if configured ---
_ed = get_subconfig("extra_datasets")
extra_datasets = _ed if isinstance(_ed, list) else []
for extra_name in extra_datasets:
    extra_cfg = datasets_config[extra_name]
    extra_df = pd.read_csv(extra_cfg["metadata_csv"])
    extra_df['sound_id'] = extra_df['sound_id'].astype(str).str.strip()
    extra_df['_audio_emb_folder'] = os.path.abspath(extra_cfg["audio_emb_folder"])
    extra_df['_text_emb_folder'] = os.path.abspath(extra_cfg["text_emb_folder"])
    extra_aux = extra_cfg.get("aux_emb_folder", None)
    extra_df['_aux_emb_folder'] = os.path.abspath(extra_aux) if extra_aux else ''
    extra_df['_source'] = extra_name
    print(f"  Extra dataset {extra_name}: {len(extra_df)} rows, {extra_df['class'].nunique()} classes")
    df = pd.concat([df, extra_df], ignore_index=True)
    print(f"  Combined total: {len(df)}")

# --- Load external mapped data (ESC-50, FSD50K) if configured ---
# External data is added AFTER the class_idx remapping (below) because
# external_mapping.csv uses remapped class_idx values while BSD10k uses
# original taxonomy codes. We defer insertion to avoid index conflicts.
_ext = get_subconfig("external_data")
_ext_rows_deferred = []  # Will be processed after class remapping
if isinstance(_ext, dict) and _ext.get("mapping_csv"):
    ext_mapping_csv = _ext["mapping_csv"]
    _ext_audio_emb_dir = os.path.abspath(_ext.get("audio_emb_folder", "data/external_embeddings/clap_audio_embeddings"))
    _ext_text_emb_dir = os.path.abspath(_ext.get("text_emb_folder", "data/external_embeddings/clap_text_embeddings"))
    ext_classes_only = _ext.get("classes_only", None)

    ext_map = pd.read_csv(ext_mapping_csv)
    print(f"\nExternal mapping: {len(ext_map)} rows from {ext_mapping_csv}")

    if ext_classes_only and isinstance(ext_classes_only, list):
        ext_map = ext_map[ext_map['bst_class'].isin(ext_classes_only)]
        print(f"  Filtered to classes {ext_classes_only}: {len(ext_map)} rows")

    for _, row in ext_map.iterrows():
        sid = str(row['source_id'])
        audio_emb_path = os.path.join(_ext_audio_emb_dir, f"{sid}.npy")
        text_emb_path = os.path.join(_ext_text_emb_dir, f"{sid}.npy")
        if os.path.isfile(audio_emb_path) and os.path.isfile(text_emb_path):
            _ext_rows_deferred.append({
                'sound_id': sid,
                'bst_class': row['bst_class'],
                'top_class': row['top_class'],
                'audio_emb_path': audio_emb_path,
                'text_emb_path': text_emb_path,
                'source': row.get('dataset_source', 'external'),
            })
    print(f"  {len(_ext_rows_deferred)} external samples with valid embeddings (deferred)")

# Discard top-level classes and classes that belong to "-other" category
s = df['class_idx'].astype(str)
df = df[~((s.str.len() == 3) & (s.str.endswith('99') | s.str.endswith('00')))].copy()
print("After filtering:", len(df))

df['original_class_idx'] = df['class_idx']

# --- Map class_idx → 0..N for training ---
original_indices = sorted(df['original_class_idx'].unique())
index_mapping = {orig: new for new, orig in enumerate(original_indices)}
df['class_idx'] = df['original_class_idx'].map(index_mapping)

# --- top class ---
df['class_top'] = df['class'].apply(lambda x: x.split('-')[0] if isinstance(x, str) else None)

df_sorted = df.sort_values('original_class_idx')

top_classes = df_sorted['class_top'].drop_duplicates()
class_top_dict = {cls: i for i, cls in enumerate(top_classes)}

df['top_class_idx'] = df['class_top'].map(class_top_dict)

# --- class dict ---
class_dict = dict(zip(df['class'], df['class_idx']))

# --- subclass dict ---
class_top_subclass_dict = {
    top_class: {
        subclass: idx
        for idx, subclass in enumerate(
            df[df['class_top'] == top_class]
            .sort_values('original_class_idx') 
            ['class']
            .drop_duplicates()
        )
    }
    for top_class in class_top_dict.keys()
}

with open(class_dict_json, 'w') as f:
    json.dump(class_dict, f, indent=4)
print(f"Saved class dictionary to {class_dict_json}")

with open(top_class_dict_json, 'w') as f:
    json.dump(class_top_dict, f, indent=4)
print(f"Saved top class dictionary to {top_class_dict_json}")

with open(top_class_subclass_dict_json, 'w') as f:
    json.dump(class_top_subclass_dict, f, indent=4)
print(f"Saved top class subclass dictionary to {top_class_subclass_dict_json}")

records = []

# --- Insert deferred external data using class_dict for correct indices ---
ext_added = 0
ext_skipped_class = 0
for ext_row in _ext_rows_deferred:
    bst_class = ext_row['bst_class']
    if bst_class not in class_dict:
        ext_skipped_class += 1
        continue  # BST class not in training set (e.g., was filtered out)
    class_idx = class_dict[bst_class]
    top_class = ext_row['top_class']
    top_class_idx = class_top_dict.get(top_class, -1)
    source = ext_row['source']

    records.append({
        "index": ext_row['sound_id'],
        "audio_emb_filepath": ext_row['audio_emb_path'],
        "text_emb_filepath": ext_row['text_emb_path'],
        "top_class": top_class,
        "top_class_idx": top_class_idx,
        "class": bst_class,
        "class_idx": class_idx,
        "confidence": float('nan'),
        "dataset_source": source,
    })
    ext_added += 1
if _ext_rows_deferred:
    print(f"\nExternal data: added {ext_added} samples, skipped {ext_skipped_class} (unmapped classes)")

for _, row in df.iterrows():
    sound_id = str(row['sound_id'])
    file = f"{sound_id}.npy"

    row_audio_folder = row.get('_audio_emb_folder', os.path.abspath(audio_emb_folder))
    row_text_folder = row.get('_text_emb_folder', os.path.abspath(text_emb_folder))

    audio_emb_filepath = os.path.join(row_audio_folder, file)
    text_emb_filepath = os.path.join(row_text_folder, file)

    if not os.path.isfile(audio_emb_filepath):
        continue

    if not os.path.isfile(text_emb_filepath):
        continue

    class_top = row['class_top']
    class_top_idx = class_top_dict.get(class_top, -1)
    class_name = row['class']
    class_idx = int(row['class_idx'])

    # Confidence and source tracking for sample weighting
    conf_val = row.get('confidence', None)
    try:
        conf_val = float(conf_val) if pd.notna(conf_val) and conf_val != '' else float('nan')
    except (ValueError, TypeError):
        conf_val = float('nan')
    source = row.get('_source', dataset_name)

    records.append({
        "index": sound_id,
        "audio_emb_filepath": audio_emb_filepath,
        "text_emb_filepath": text_emb_filepath,
        "top_class": class_top,
        "top_class_idx": class_top_idx,
        "class": class_name,
        "class_idx": class_idx,
        "confidence": conf_val,
        "dataset_source": source,
    })
    # Add auxiliary embedding path if configured
    row_aux_folder = row.get('_aux_emb_folder', '')
    if not row_aux_folder and aux_emb_folder:
        row_aux_folder = os.path.abspath(aux_emb_folder)
    if row_aux_folder:
        aux_emb_filepath = os.path.join(row_aux_folder, file)
        if os.path.isfile(aux_emb_filepath):
            records[-1]["aux_emb_filepath"] = aux_emb_filepath

db_df = pd.DataFrame(records)
db_df.to_csv(processed_dataset_csv, index=False)
print(f"Saved embedding dataframe to {processed_dataset_csv}")
print(f"Dataset built with {len(db_df)} samples.")

# Print source and class distribution
if 'dataset_source' in db_df.columns:
    print(f"\n=== Dataset Source Distribution ===")
    for src, cnt in db_df['dataset_source'].value_counts().items():
        print(f"  {src}: {cnt} samples")
    print(f"\n=== Class Distribution (top 10) ===")
    for cls, cnt in db_df['class'].value_counts().head(10).items():
        print(f"  {cls}: {cnt}")