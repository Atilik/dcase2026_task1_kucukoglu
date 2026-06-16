"""Build combined dataset.csv merging BSD10k and external embeddings."""
import os
import pandas as pd

BSD10K_PROCESSED = "data/processed_dataset.csv"
EXTERNAL_MAPPING = "data/external_mapping.csv"
COMBINED_OUTPUT = "data/combined_dataset.csv"

BASE_DIR = "/scratch/mk9649/repos/dcase2026_task1_baseline"

def main():
    print("Loading datasets...")
    bsd10k_df = pd.read_csv(BSD10K_PROCESSED)
    ext_df = pd.read_csv(EXTERNAL_MAPPING)
    
    print(f"BSD10k samples: {len(bsd10k_df)}")
    print(f"External samples: {len(ext_df)}")
    
    # Format external to match BSD10k
    # BSD10k columns: index,audio_emb_filepath,text_emb_filepath,top_class,top_class_idx,class,class_idx,confidence,dataset_source
    
    formatted_ext = pd.DataFrame()
    formatted_ext['index'] = ext_df['source_id']
    
    # Generate full paths for embeddings
    formatted_ext['audio_emb_filepath'] = ext_df['source_id'].apply(
        lambda x: os.path.join(BASE_DIR, f"data/external_embeddings/clap_audio_embeddings/{x}.npy")
    )
    formatted_ext['text_emb_filepath'] = ext_df['source_id'].apply(
        lambda x: os.path.join(BASE_DIR, f"data/external_embeddings/clap_text_embeddings/{x}.npy")
    )
    
    formatted_ext['top_class'] = ext_df['top_class']
    formatted_ext['top_class_idx'] = ext_df['top_class_idx']
    formatted_ext['class'] = ext_df['bst_class']
    formatted_ext['class_idx'] = ext_df['class_idx']
    
    # Set confidence to NaN for external data (no confidence annotation available)
    # NaN is handled correctly by confidence filtering (NaN samples are always kept)
    formatted_ext['confidence'] = float('nan')
    
    formatted_ext['dataset_source'] = ext_df['dataset_source']
    
    # Verify files exist
    print("\nVerifying external embedding files exist...")
    missing_audio = 0
    missing_text = 0
    
    for _, row in formatted_ext.iterrows():
        if not os.path.exists(row['audio_emb_filepath']):
            missing_audio += 1
        if not os.path.exists(row['text_emb_filepath']):
            missing_text += 1
            
    if missing_audio > 0 or missing_text > 0:
        print(f"WARNING: Missing {missing_audio} audio embeddings and {missing_text} text embeddings.")
    else:
        print("All external embedding files found!")
        
    # Combine ALL
    combined_all = pd.concat([bsd10k_df, formatted_ext], ignore_index=True)
    combined_all.to_csv("data/combined_dataset_all.csv", index=False)
    print(f"\nSaved combined ALL dataset to data/combined_dataset_all.csv")
    print(f"Total samples: {len(combined_all)}")
    
    # Combine WEAK only
    weak_classes = ['sp-c', 'ss-i', 'fx-ex']
    weak_ext = formatted_ext[formatted_ext['class'].isin(weak_classes)]
    combined_weak = pd.concat([bsd10k_df, weak_ext], ignore_index=True)
    combined_weak.to_csv("data/combined_dataset_weak.csv", index=False)
    print(f"\nSaved combined WEAK dataset to data/combined_dataset_weak.csv")
    print(f"Total samples: {len(combined_weak)}")

if __name__ == "__main__":
    main()
