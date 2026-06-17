import random
import collections.abc
from collections import defaultdict
import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from losses import CrossEntropyLoss, SupervisedContrastiveLoss, compute_class_weights
from utils import get_subconfig, set_seed, build_class_to_topclass_mapping, build_class_to_topclass_tensor
from models import BaseClassifier
from dataset_utils import HATRDataset
from evaluate import evaluate_model

# Paths
dataset_name = get_subconfig("active_dataset")
dataset_path = get_subconfig("datasets")[dataset_name]["metadata_csv"]
color_dict_path = get_subconfig("color_dict_path")
top_color_dict_path = get_subconfig("top_color_dict_path")

data_dir = get_subconfig("output_path")
prepared_dataset_path = os.path.join(data_dir, get_subconfig("processed_dataset_csv"))
class_dict_json = os.path.join(data_dir, get_subconfig("class_dict_json"))
top_class_dict_json = os.path.join(data_dir, get_subconfig("top_class_dict_json"))
subclass_json = os.path.join(data_dir, get_subconfig("top_class_subclass_dict_json"))


def init_weights(model):
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, mode='fan_out')
    elif isinstance(model, nn.Linear):
        nn.init.xavier_uniform_(model.weight)

def make_serializable(obj, decimals=6):
    """Recursively convert tensors, numpy arrays, and numbers to JSON-serializable types with rounding."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
        return make_serializable(obj, decimals)
    elif isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return round(float(obj), decimals)
        else:
            return [make_serializable(x, decimals) for x in obj]
    elif isinstance(obj, float):
        return round(obj, decimals)
    elif isinstance(obj, int):
        return obj
    elif isinstance(obj, collections.abc.Mapping):
        return {k: make_serializable(v, decimals) for k, v in obj.items()}
    elif isinstance(obj, collections.abc.Iterable) and not isinstance(obj, (str, bytes)):
        return [make_serializable(x, decimals) for x in obj]
    else:
        return obj
    
def train_model(model, train_loader, val_loader, device,
                num_epochs=100, lr=0.001, classification_weight=1.0, classification_criterion=None, 
                output_dir='model_outputs/model_output', scheduler_type='plateau', patience=10, early_stopping_factor=5,
                top_class_loss_weight=0.0, class_to_topclass_tensor=None, mixup_alpha=0.0,
                cutmix_alpha=0.0, label_smoothing=0.0,
                contrastive_criterion=None, contrastive_lambda=0.0):
    """
    Train a model with validation, LR scheduling, checkpointing, and early stopping.

    Tracks training loss, validation accuracy, and (if available) attention statistics.
    Saves the best model with a config, and training history to `output_dir`.

    Returns:
        best_accuracy (float), history (dict), model (nn.Module)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    if scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=patience, verbose=True)
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    else:
        scheduler = None
    
    best_accuracy = 0.0
    epochs_without_improvement = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        losses = defaultdict(float)
        total_samples = 0

        attn_audio_epoch = []
        attn_text_epoch = []

        for data in train_loader:
            class_labels = data['class_idx'].to(device)
            audio_emb = data.get('audio_embedding', None)
            text_emb = data.get('text_embedding', None)
            aux_emb = data.get('aux_embedding', None)

            batch_size = class_labels.size(0)
            total_samples += batch_size
            
            if audio_emb is not None:
                audio_emb = audio_emb.to(device)
            if text_emb is not None:
                text_emb = text_emb.to(device)
            if aux_emb is not None:
                aux_emb = aux_emb.to(device)

            optimizer.zero_grad()

            # Mixup augmentation
            if mixup_alpha > 0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                index = torch.randperm(batch_size).to(device)
                if audio_emb is not None:
                    audio_emb = lam * audio_emb + (1 - lam) * audio_emb[index]
                if text_emb is not None:
                    text_emb = lam * text_emb + (1 - lam) * text_emb[index]
                if aux_emb is not None:
                    aux_emb = lam * aux_emb + (1 - lam) * aux_emb[index]

            # CutMix augmentation (swap contiguous embedding dimensions)
            if cutmix_alpha > 0 and mixup_alpha == 0:
                lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                index = torch.randperm(batch_size).to(device)
                emb_dim = audio_emb.shape[1] if audio_emb is not None else text_emb.shape[1]
                cut_len = int(emb_dim * (1 - lam))
                cut_start = random.randint(0, emb_dim - cut_len)
                cut_end = cut_start + cut_len
                if audio_emb is not None:
                    audio_emb[:, cut_start:cut_end] = audio_emb[index, cut_start:cut_end]
                if text_emb is not None:
                    text_emb[:, cut_start:cut_end] = text_emb[index, cut_start:cut_end]
                if aux_emb is not None:
                    aux_emb[:, cut_start:cut_end] = aux_emb[index, cut_start:cut_end]

            z, class_logit, top_class_logit, attn_scores = model(audio_emb, text_emb, aux_emb)
            
            # collect batch attention once per batch
            if attn_scores is not None:
                attn_audio_epoch.append(attn_scores[:, 0].detach().cpu())
                attn_text_epoch.append(attn_scores[:, 1].detach().cpu())

            total_loss = 0.0

            if classification_criterion is not None:
                sample_weights = data.get('sample_weight', None)
                if sample_weights is not None:
                    sample_weights = sample_weights.to(device)

                if mixup_alpha > 0:
                    # Mixup loss: blend losses from both labels
                    loss_a = classification_criterion(class_logit, class_labels, sample_weights)
                    loss_b = classification_criterion(class_logit, class_labels[index], sample_weights)
                    cls_loss = lam * loss_a + (1 - lam) * loss_b
                elif cutmix_alpha > 0:
                    # CutMix loss: blend like mixup
                    loss_a = classification_criterion(class_logit, class_labels, sample_weights)
                    loss_b = classification_criterion(class_logit, class_labels[index], sample_weights)
                    cls_loss = lam * loss_a + (1 - lam) * loss_b
                else:
                    cls_loss = classification_criterion(class_logit, class_labels, sample_weights)

                losses['cls'] += cls_loss.item() * batch_size
                total_loss += classification_weight * cls_loss

            # Hierarchical top-class auxiliary loss
            if top_class_logit is not None and top_class_loss_weight > 0 and class_to_topclass_tensor is not None:
                top_class_labels = class_to_topclass_tensor[class_labels]
                if mixup_alpha > 0:
                    top_loss_a = classification_criterion(top_class_logit, top_class_labels)
                    top_loss_b = classification_criterion(top_class_logit, class_to_topclass_tensor[class_labels[index]])
                    top_cls_loss = lam * top_loss_a + (1 - lam) * top_loss_b
                else:
                    top_cls_loss = classification_criterion(top_class_logit, top_class_labels)
                losses['top_cls'] += top_cls_loss.item() * batch_size
                total_loss += top_class_loss_weight * top_cls_loss

            # Supervised contrastive loss on latent z using top-class labels
            if contrastive_criterion is not None and contrastive_lambda > 0:
                top_class_labels = data['top_class_idx'].to(device)
                contrastive_loss = contrastive_criterion(z, top_class_labels)
                losses['contrastive'] = losses.get('contrastive', 0) + contrastive_loss.item() * batch_size
                total_loss += contrastive_lambda * contrastive_loss

            total_loss.backward()
            optimizer.step()
            losses['total'] += total_loss.item() * batch_size

        # per-epoch attention summary
        if attn_audio_epoch:
            attn_audio_epoch = torch.cat(attn_audio_epoch, dim=0)
            attn_text_epoch = torch.cat(attn_text_epoch, dim=0)
            history["attention_audio"].append(attn_audio_epoch.mean(0).numpy())
            history["attention_text"].append(attn_text_epoch.mean(0).numpy())

        num_batches = len(train_loader)
        for k in losses:
            history[f'train_{k}_loss'].append(losses[k] / total_samples)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_loader:
                labels = data['class_idx'].to(device)
                audio_emb = data.get('audio_embedding', None)
                text_emb = data.get('text_embedding', None)
                aux_emb = data.get('aux_embedding', None)
                
                if audio_emb is not None:
                    audio_emb = audio_emb.to(device)
                if text_emb is not None:
                    text_emb = text_emb.to(device)
                if aux_emb is not None:
                    aux_emb = aux_emb.to(device)

                _, class_logit, _, _ = model(audio_emb, text_emb, aux_emb)
                    
                _, predicted = torch.max(class_logit.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        history['val_accuracy'].append(val_accuracy)

        with open(os.path.join(output_dir, "history.json"), "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Val acc: {val_accuracy:.2f}%")
        # for k in losses:
        #     if losses[k] > 0:
        #         print(f"  {k.capitalize()} loss: {losses[k] / total_samples:.4f}")
        # print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if scheduler:
            if scheduler_type == 'plateau':
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model_config = {'hidden_size': hidden_size, 'num_classes': len(class_dict),
                'emb_size_audio': emb_size_audio, 'emb_size_text': emb_size_text, 
                'dropout': dropout, 'use_batch_norm': True,'mode': mode,
                'num_top_classes': num_top_classes,
                'emb_size_aux': emb_size_aux,
            }

            torch.save({
                'model_state': model.state_dict(),
                'config': model_config,
            }, os.path.join(output_dir, "best_model.pth"))

            print(f"  New best model saved")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience * early_stopping_factor:
                print("Early stopping triggered.")
                break

    return best_accuracy, history, model


if __name__ == "__main__":
    seed = set_seed()  # For reproducibility

    with open(class_dict_json, 'r') as f:
        class_dict = json.load(f)
    with open(top_class_dict_json, 'r') as f:
        top_class_dict = json.load(f)

    modes = get_subconfig('modes') or ['both', 'audio']  # Read from config, default to both+audio
    model_output = get_subconfig('model_output_dir') or './model_outputs/model_output'  # Read from config, default to model_output

    batch_size = 64
    num_epochs = 100
    learning_rate = 0.001
    classification_weight = 1
    scheduler_type = 'step'
    patience = 5
    early_stopping_factor = 3
    k_folds = 5

    full_df = pd.read_csv(prepared_dataset_path)

    # Load confidence scores from original metadata and merge
    _ct = get_subconfig('confidence_threshold')
    confidence_threshold = _ct if isinstance(_ct, (int, float)) else 0
    if confidence_threshold > 0:
        meta_df = pd.read_csv(dataset_path)
        meta_df['sound_id'] = meta_df['sound_id'].astype(str).str.strip()
        conf_map = meta_df.set_index('sound_id')['confidence']
        full_df['confidence'] = full_df['index'].astype(str).map(conf_map)
        print(f"Confidence filtering enabled: threshold >= {confidence_threshold}")
        print(f"  Samples with confidence data: {full_df['confidence'].notna().sum()} / {len(full_df)}")

    # Sample weighting config
    _cw = get_subconfig('use_confidence_weighting')
    use_confidence_weighting = _cw if isinstance(_cw, bool) else False
    _dwm = get_subconfig('dataset_weight_map')
    dataset_weight_map = _dwm if isinstance(_dwm, dict) and _dwm else {}
    _cwfn = get_subconfig('confidence_weight_fn')
    confidence_weight_fn = _cwfn if isinstance(_cwfn, str) else 'linear'
    _ucw = get_subconfig('use_class_weights')
    use_class_weights = _ucw if isinstance(_ucw, bool) else False
    _cwm = get_subconfig('class_weight_method')
    class_weight_method = _cwm if isinstance(_cwm, str) else 'inverse_sqrt'
    if use_confidence_weighting or dataset_weight_map:
        print(f"Sample weighting: confidence_weighting={use_confidence_weighting}, fn={confidence_weight_fn}, dataset_weights={dataset_weight_map}")
    if use_class_weights:
        print(f"Class weighting enabled: method={class_weight_method}")

    # Mixup config
    _ma = get_subconfig('mixup_alpha')
    mixup_alpha = float(_ma) if isinstance(_ma, (int, float)) and _ma > 0 else 0.0
    if mixup_alpha > 0:
        print(f"Mixup augmentation enabled: alpha={mixup_alpha}")

    # CutMix config
    _cma = get_subconfig('cutmix_alpha')
    cutmix_alpha = float(_cma) if isinstance(_cma, (int, float)) and _cma > 0 else 0.0
    if cutmix_alpha > 0:
        print(f"CutMix augmentation enabled: alpha={cutmix_alpha}")

    # Label smoothing config
    _ls = get_subconfig('label_smoothing')
    label_smoothing = float(_ls) if isinstance(_ls, (int, float)) and _ls > 0 else 0.0
    if label_smoothing > 0:
        print(f"Label smoothing enabled: {label_smoothing}")

    # Noise std config
    _ns = get_subconfig('noise_std')
    noise_std = float(_ns) if isinstance(_ns, (int, float)) and _ns > 0 else 0.0001

    # Cross-modal swap config
    _cms = get_subconfig('cross_modal_swap_prob')
    cross_modal_swap_prob = float(_cms) if isinstance(_cms, (int, float)) and _cms > 0 else 0.0
    if cross_modal_swap_prob > 0:
        print(f"Cross-modal swap enabled: prob={cross_modal_swap_prob}")

    # Contrastive loss config
    _ucl = get_subconfig('use_contrastive_loss')
    use_contrastive_loss = _ucl if isinstance(_ucl, bool) else False
    _ct = get_subconfig('contrastive_temperature')
    contrastive_temperature = float(_ct) if isinstance(_ct, (int, float)) and _ct > 0 else 0.5
    _cl = get_subconfig('contrastive_lambda')
    contrastive_lambda = float(_cl) if isinstance(_cl, (int, float)) and _cl > 0 else 1.0
    if use_contrastive_loss:
        print(f"Supervised contrastive loss enabled: λ={contrastive_lambda}, τ={contrastive_temperature}")

    # Class equalization config (balanced BSD35k sampling)
    _uce = get_subconfig('use_class_equalization')
    use_class_equalization = _uce if isinstance(_uce, bool) else False
    _et = get_subconfig('equalization_target')
    equalization_target = _et if isinstance(_et, str) else 'max'
    _ef = get_subconfig('equalization_factor')
    equalization_factor = float(_ef) if isinstance(_ef, (int, float)) else 1.0
    _os = get_subconfig('oversample')
    oversample = _os if isinstance(_os, bool) else True
    if use_class_equalization:
        print(f"Class equalization enabled: target={equalization_target}, factor={equalization_factor}, oversample={oversample}")

    datasets = {
        f'{dataset_name} full': {'df': full_df}
    }

    for dataset, dataset_info in datasets.items():
        print(f"\n=== Dataset: {dataset} ===")
        database = dataset_info['df']

        # Separate primary (BSD10k) and extra (BSD35k-CS) samples
        # K-fold split only on primary dataset; extra goes to training only
        if 'dataset_source' in database.columns:
            primary_df = database[database['dataset_source'] == dataset_name].reset_index(drop=True)
            extra_df = database[database['dataset_source'] != dataset_name].reset_index(drop=True)
            if len(extra_df) > 0:
                print(f"  Primary ({dataset_name}): {len(primary_df)} samples")
                print(f"  Extra (training only): {len(extra_df)} samples")
        else:
            primary_df = database
            extra_df = pd.DataFrame()

        labels = primary_df["class_idx"].tolist()

        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)

        for mode in modes:
            print(f"\n=== Running experiments: Dataset={dataset} | Mode={mode} ===")

            for fold, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                print(f"\n==== Fold {fold} ====")

                trainval_labels = [labels[i] for i in trainval_idx]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
                train_idx_rel, val_idx_rel = next(sss.split(np.zeros(len(trainval_labels)), trainval_labels))
                train_idx = [trainval_idx[i] for i in train_idx_rel]
                val_idx = [trainval_idx[i] for i in val_idx_rel]

                train_df = primary_df.iloc[train_idx].reset_index(drop=True)
                val_df = primary_df.iloc[val_idx].reset_index(drop=True)
                test_df = primary_df.iloc[test_idx].reset_index(drop=True)

                # Per-fold embedding swapping: if embeddings have fold_X/ subdirs,
                # swap paths so each fold uses its own fine-tuned CLAP checkpoint's
                # embeddings (prevents data leakage from fine-tuning)
                if 'audio_emb_filepath' in train_df.columns and len(train_df) > 0:
                    sample_path = train_df['audio_emb_filepath'].iloc[0]
                    parent_dir = os.path.dirname(sample_path)
                    fold_dir = os.path.join(parent_dir, f"fold_{fold}")
                    if os.path.isdir(fold_dir):
                        print(f"  Using per-fold embeddings from {fold_dir}")
                        def swap_to_fold(path, fold_dir=fold_dir):
                            basename = os.path.basename(path)
                            return os.path.join(fold_dir, basename)
                        train_df = train_df.copy()
                        val_df = val_df.copy()
                        test_df = test_df.copy()
                        train_df['audio_emb_filepath'] = train_df['audio_emb_filepath'].apply(swap_to_fold)
                        val_df['audio_emb_filepath'] = val_df['audio_emb_filepath'].apply(swap_to_fold)
                        test_df['audio_emb_filepath'] = test_df['audio_emb_filepath'].apply(swap_to_fold)

                # Add extra dataset (BSD35k-CS) to training only
                if len(extra_df) > 0:
                    if use_class_equalization:
                        # Balanced: only add BSD35k samples to equalize class counts
                        bsd10k_counts = train_df['class'].value_counts()
                        if equalization_target == 'max':
                            target_count = int(bsd10k_counts.max() * equalization_factor)
                        elif equalization_target == 'median':
                            target_count = int(bsd10k_counts.median() * equalization_factor)
                        else:
                            target_count = int(equalization_target) if str(equalization_target).isdigit() else int(bsd10k_counts.max())
                        
                        extra_sampled = []
                        for cls_name in bsd10k_counts.index:
                            current = bsd10k_counts[cls_name]
                            needed = max(0, target_count - current)
                            if needed > 0:
                                cls_pool = extra_df[extra_df['class'] == cls_name]
                                if len(cls_pool) > 0:
                                    n_sample = min(needed, len(cls_pool)) if not oversample else needed
                                    sampled = cls_pool.sample(n=n_sample, replace=(n_sample > len(cls_pool)), random_state=42)
                                    extra_sampled.append(sampled)
                        
                        if extra_sampled:
                            balanced_extra = pd.concat(extra_sampled, ignore_index=True)
                            train_df = pd.concat([train_df, balanced_extra], ignore_index=True)
                            print(f"  Balanced equalization: added {len(balanced_extra)} samples (target={target_count}/class)")
                        else:
                            print(f"  Balanced equalization: no samples needed")
                    else:
                        train_df = pd.concat([train_df, extra_df], ignore_index=True)
                        print(f"  Added {len(extra_df)} extra samples to training")

                # Apply confidence filtering to train/val ONLY (test stays full)
                # Samples with NaN confidence (e.g. BSD35k-CS) are kept
                if confidence_threshold > 0 and 'confidence' in train_df.columns:
                    train_before = len(train_df)
                    val_before = len(val_df)
                    train_df = train_df[(train_df['confidence'].isna()) | (train_df['confidence'] >= confidence_threshold)].reset_index(drop=True)
                    val_df = val_df[(val_df['confidence'].isna()) | (val_df['confidence'] >= confidence_threshold)].reset_index(drop=True)
                    print(f"  Confidence filter: train {train_before}->{len(train_df)}, val {val_before}->{len(val_df)}, test {len(test_df)} (unchanged)")

                print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")

                train_dataset = HATRDataset(train_df, aug=True, mask_pct=0.7,
                    use_confidence_weighting=use_confidence_weighting,
                    dataset_weight_map=dataset_weight_map,
                    confidence_weight_fn=confidence_weight_fn,
                    noise_std=noise_std,
                    cross_modal_swap_prob=cross_modal_swap_prob)
                val_dataset = HATRDataset(val_df, aug=False)
                test_dataset = HATRDataset(test_df, aug=False)

                # num_workers=4 deadlocks inside Singularity with GPU; use 0 for GPU, 4 for CPU
                num_workers = 0 if torch.cuda.is_available() else 4

                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=torch.cuda.is_available()
                )

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                _cfg_emb_audio = get_subconfig('emb_size_audio') or 512
                _cfg_emb_text = get_subconfig('emb_size_text') or 512
                _cfg_emb_aux = get_subconfig('emb_size_aux') or 0
                emb_size_audio = _cfg_emb_audio if mode in ['audio', 'both'] else 0
                emb_size_text = _cfg_emb_text if mode in ['text', 'both'] else 0
                emb_size_aux = int(_cfg_emb_aux) if mode == 'both' else 0

                hidden_size = 128
                dropout = 0.1
                use_batch_norm = True

                # Read hierarchical loss settings from config
                top_class_loss_weight = get_subconfig('top_class_loss_weight') or 0.0
                num_top_classes = len(top_class_dict) if top_class_loss_weight > 0 else 0

                model = BaseClassifier(
                    hidden_size=128,
                    num_classes=len(class_dict),
                    emb_size_audio=emb_size_audio,
                    emb_size_text=emb_size_text,
                    dropout=dropout,
                    use_batch_norm=use_batch_norm,
                    mode=mode,
                    num_top_classes=num_top_classes,
                    emb_size_aux=emb_size_aux,
                ).to(device)

                # Class-weighted loss
                if use_class_weights:
                    cw = compute_class_weights(train_df, len(class_dict), method=class_weight_method).to(device)
                    print(f"  Class weights (min={cw.min():.3f}, max={cw.max():.3f}, mean={cw.mean():.3f})")
                    classification_criterion = CrossEntropyLoss(class_weights=cw, label_smoothing=label_smoothing)
                else:
                    classification_criterion = CrossEntropyLoss(label_smoothing=label_smoothing)

                # Build class-to-topclass mapping tensor for hierarchical loss
                subclass_to_topclass_tensor = build_class_to_topclass_tensor(class_dict, top_class_dict, device) if num_top_classes > 0 else None

                # Contrastive loss criterion
                contrastive_criterion = SupervisedContrastiveLoss(temperature=contrastive_temperature) if use_contrastive_loss else None

                output_dir = os.path.join(
                    model_output,
                    mode, f"fold_{fold}"
                )
                os.makedirs(output_dir, exist_ok=True)

                model_path = os.path.join(output_dir, "best_model.pth")

                init_weights(model)

                best_accuracy, history, trained_model = train_model(
                    model, train_loader, val_loader, device,
                    num_epochs=num_epochs, lr=learning_rate,
                    classification_weight=classification_weight,
                    classification_criterion=classification_criterion,
                    output_dir=output_dir,
                    scheduler_type=scheduler_type, patience=patience, early_stopping_factor=early_stopping_factor,
                    top_class_loss_weight=top_class_loss_weight,
                    class_to_topclass_tensor=subclass_to_topclass_tensor,
                    mixup_alpha=mixup_alpha,
                    cutmix_alpha=cutmix_alpha,
                    label_smoothing=label_smoothing,
                    contrastive_criterion=contrastive_criterion,
                    contrastive_lambda=contrastive_lambda if use_contrastive_loss else 0.0
                )
                print(f"Best validation accuracy: {best_accuracy:.2f}%")

                # Save splits for reproducibility
                splits_df = pd.concat([
                    train_df[['index']].assign(split='train'),
                    val_df[['index']].assign(split='val'),
                    test_df[['index']].assign(split='test')
                ])
                splits_df.to_csv(os.path.join(output_dir, "splits.csv"), index=False)

                # Save updated history with model info
                history['model_info'] = {
                    'model_class': trained_model.__class__.__name__,
                    'hidden_size': hidden_size,
                    'num_classes': len(class_dict),
                    'emb_size_audio': emb_size_audio,
                    'emb_size_text': emb_size_text,
                    'dropout': dropout,
                    'use_batch_norm': True,
                    'mode': mode,
                    'num_folds': k_folds,
                    'fold_id': fold,
                    'batch_size': batch_size,
                    'random_seed': seed,
                }
                
                history_path = os.path.join(output_dir, "history.json")
                with open(history_path, "w") as f:
                    json.dump(make_serializable(history), f, indent=2)

                # Testing
                class_to_top_class = build_class_to_topclass_mapping(class_dict, top_class_dict)
                subclass_to_topclass_tensor = build_class_to_topclass_tensor(class_dict, top_class_dict, device)

                metrics = evaluate_model(
                    BaseClassifier,
                    model_path,
                    test_loader,
                    device,
                    class_to_top_class,
                    output_dir=output_dir,
                    fold_id=fold,
                    class_dict=class_dict,
                )

                print("\n===== Fold Results =====")
                print(f"Final model accuracy: {metrics['accuracy']:.2f}%")
                print(f"Final model top-level accuracy: {metrics['top_accuracy']:.2f}%")
                print("========================")

    print("All experiments done!")
