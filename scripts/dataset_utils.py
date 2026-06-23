import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

_emb_cache = {}

def _load_npy_cached(path):
    if not isinstance(path, str) or str(path) == 'nan' or not path:
        return None
        
    # Map old hardcoded baseline repo path to actual repository root path
    if "dcase2026_task1_baseline" in path:
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
        path = path.replace("/scratch/mk9649/repos/dcase2026_task1_baseline", project_root)
        
    if path not in _emb_cache:
        try:
            _emb_cache[path] = torch.tensor(np.load(path), dtype=torch.float32)
        except Exception as e:
            print(f"Error loading embedding from path {path}: {e}")
            return None
    return _emb_cache[path].clone()

class HATRDataset(Dataset):
    """
    Dataset for precomputed multimodal (audio + text) embeddings.
    Hierarchical labels (2-level, use only parent and leafs if more).
    Augmentation (optional): Gaussian noise + random zeroing on embeddings.
    CSV columns are hardcoded for BST datasets, change if necessary.
    """

    def __init__(self, dataframe, aug=True, mask_pct=0.2,
                 use_confidence_weighting=False, dataset_weight_map=None,
                 confidence_weight_fn='linear', noise_std=0.0001,
                 cross_modal_swap_prob=0.0):
        self.dataframe = dataframe
        self.aug = aug
        self.mask_pct = mask_pct
        self.use_confidence_weighting = use_confidence_weighting
        self.dataset_weight_map = dataset_weight_map or {}
        self.confidence_weight_fn = confidence_weight_fn  # 'linear', 'shifted', 'binary'
        self.noise_std = noise_std
        self.cross_modal_swap_prob = cross_modal_swap_prob
        
        # Build class-to-indices mapping for cross-modal swap
        if cross_modal_swap_prob > 0:
            self._class_indices = {}
            for i, row in self.dataframe.iterrows():
                cls = row['class']
                if cls not in self._class_indices:
                    self._class_indices[cls] = []
                self._class_indices[cls].append(i)

    def _rand_mask(self, emb):
        max_mask = int(emb.shape[0] * self.mask_pct)
        num_to_mask = random.randint(1, max_mask)
        mask_indices = torch.randperm(emb.shape[0])[:num_to_mask]
        mask = torch.ones_like(emb)
        mask[mask_indices] = 0.0
        return emb * mask

    def _add_noise(self, emb):
        return emb + torch.randn_like(emb) * self.noise_std

    def get_classes(self):
        return self.dataframe['class'].unique()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        sample = self.dataframe.iloc[idx]
        sound_id = sample['index']
        class_name = sample['class']
        top_class_name = sample['top_class']
        class_idx = sample['class_idx']
        top_class_idx = sample['top_class_idx']
        
        emb_path = sample['audio_emb_filepath']
        emb = _load_npy_cached(emb_path)

        text_path = sample['text_emb_filepath']
        text_emb = _load_npy_cached(text_path)

        # Optional auxiliary embedding (e.g., Whisper, ConvNeXt)
        aux_emb = None
        if 'aux_emb_filepath' in sample.index:
            aux_emb = _load_npy_cached(sample['aux_emb_filepath'])

        if self.aug:
            # Cross-modal swap: replace text with another same-class sample's text
            if self.cross_modal_swap_prob > 0 and random.random() < self.cross_modal_swap_prob:
                same_class = self._class_indices.get(class_name, [])
                if len(same_class) > 1:
                    swap_idx = random.choice(same_class)
                    swap_sample = self.dataframe.iloc[swap_idx]
                    swap_text_path = swap_sample['text_emb_filepath']
                    text_emb = _load_npy_cached(swap_text_path)

            # Gaussian noise
            emb = self._add_noise(emb)
            text_emb = self._add_noise(text_emb)
            if aux_emb is not None:
                aux_emb = self._add_noise(aux_emb)

            # Random masking
            emb = self._rand_mask(emb)
            text_emb = self._rand_mask(text_emb)
            if aux_emb is not None:
                aux_emb = self._rand_mask(aux_emb)

        # Compute sample weight
        weight = 1.0
        source = sample.get('dataset_source', None)
        if source and source in self.dataset_weight_map:
            weight = self.dataset_weight_map[source]
        if self.use_confidence_weighting and 'confidence' in sample.index:
            conf = sample['confidence']
            try:
                conf = float(conf)
                if not np.isnan(conf):
                    if self.confidence_weight_fn == 'linear':
                        weight = conf / 5.0           # conf 1→0.2, 5→1.0
                    elif self.confidence_weight_fn == 'shifted':
                        weight = (conf - 1.0) / 4.0   # conf 1→0.0, 5→1.0
                    elif self.confidence_weight_fn == 'binary':
                        weight = 1.0 if conf >= 3 else 0.3
                    else:
                        weight = conf / 5.0  # fallback to linear
            except (ValueError, TypeError):
                pass  # keep default weight

        sample_data = {
            'sound_id': sound_id,
            'audio_embedding': emb,
            'text_embedding': text_emb,
            'class': class_name,
            'class_idx': class_idx,
            'top_class': top_class_name,
            'top_class_idx': top_class_idx,
            'sample_weight': torch.tensor(weight, dtype=torch.float32),
        }
        if aux_emb is not None:
            sample_data['aux_embedding'] = aux_emb
        
        return sample_data
