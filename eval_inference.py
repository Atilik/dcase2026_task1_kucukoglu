"""Eval inference for ALL 4 DCASE 2026 Task 1 submissions.

Produces:
  Kucukoglu_NYU_task1_1.output.csv  (5-model multi-encoder ensemble)
  Kucukoglu_NYU_task1_2.output.csv  (5-model CLAP+ConvNeXt ensemble)
  Kucukoglu_NYU_task1_3.output.csv  (single model: xswap)
  Kucukoglu_NYU_task1_4.output.csv  (single model: 3mod_mx02)

Usage:
    ./sing python -u eval_inference.py          # full pipeline
    ./sing python -u eval_inference.py --skip-extraction  # embeddings already cached
    ./sing python -u eval_inference.py --sub 1  # only submission 1
"""
import os, sys, re, json, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import soundfile as sf
import librosa

from models import BaseClassifier
from utils import build_id_to_class_mapping

# ── Paths ──
EVAL_AUDIO_DIR = "data/eval/audio"
EVAL_METADATA = "data/eval/metadata.csv"
EVAL_CLAP_AUDIO_DIR = "data/eval/features/clap_audio_embeddings"
EVAL_CLAP_TEXT_DIR = "data/eval/features/clap_text_embeddings"
EVAL_CONVNEXT_DIR = "data/eval/features/convnext_audio_embeddings"
EVAL_WHISPER_DIR = "data/eval/features/whisper_audio_embeddings"
CLASS_DICT_PATH = "data/class_dict.json"
TEAM = "Kucukoglu_NYU"
NUM_FOLDS = 5

# ── 4 Submission configs ──
# Each: (sub_id, name, [(model_name, model_dir, audio_type, aux_type)])
# audio_type: "clap" | "convnext"
# aux_type: None | "whisper"
SUBMISSIONS = {
    1: {
        "name": "5-model multi-encoder ensemble",
        "models": [
            ("mixup03",   "model_output_mixup03/both",          "clap",     None),
            ("m02_hl07",  "model_output_mixup02_hloss07/both",  "clap",     None),
            ("bal_med",   "model_output_balanced_med/both",      "clap",     None),
            ("cnx_mix02", "model_output_convnext_mix02/both",   "convnext", None),
        ],
        "clap_ft": True,  # also include clap_ft logits
    },
    2: {
        "name": "5-model CLAP+ConvNeXt ensemble",
        "models": [
            ("hloss_070", "model_output_clap_hloss_w070/both",  "clap",     None),
            ("m02_clw",   "model_output_mixup02_classw/both",   "clap",     None),
            ("bal_med",   "model_output_balanced_med/both",      "clap",     None),
            ("combo_aug", "model_output_combo_aug/both",         "clap",     None),
            ("cnx_mix02", "model_output_convnext_mix02/both",   "convnext", None),
        ],
        "clap_ft": False,
    },
    3: {
        "name": "Single model: xswap",
        "models": [
            ("xswap", "model_output_xswap_noise/both", "clap", None),
        ],
        "clap_ft": False,
    },
    4: {
        "name": "Single model: 3mod_mx02 (CLAP+Whisper)",
        "models": [
            ("3mod_mx02", "model_output_3mod_mix02/both", "clap", "whisper"),
        ],
        "clap_ft": False,
    },
}


# ═══════════════════════════════════════════════════════════════
# STEP 1: EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════

def clean_text(title, tags, description, max_chars=400):
    parts = []
    if pd.notna(title) and str(title).strip():
        parts.append(str(title).strip())
    if pd.notna(tags) and str(tags).strip():
        parts.append(str(tags).strip().replace(",", " "))
    if pd.notna(description) and str(description).strip():
        desc = re.sub(r'<[^>]+>', ' ', str(description).strip())
        desc = re.sub(r'\s+', ' ', desc).strip()
        parts.append(desc)
    text = ". ".join(parts) if parts else "audio recording"
    return text[:max_chars] if len(text) > max_chars else text


def extract_clap_embeddings(device):
    """Extract CLAP audio + text embeddings for eval set."""
    from huggingface_hub import hf_hub_download
    import laion_clap

    eval_df = pd.read_csv(EVAL_METADATA)
    sample_ids = eval_df['anonymous_id'].tolist()
    print(f"\n{'='*60}\nCLAP EMBEDDING EXTRACTION ({len(sample_ids)} samples)\n{'='*60}")

    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module.load_ckpt(clap_ckpt)
    clap_module.to(device)
    clap_module.eval()

    # Audio embeddings
    os.makedirs(EVAL_CLAP_AUDIO_DIR, exist_ok=True)
    to_extract = [s for s in sample_ids if not os.path.exists(os.path.join(EVAL_CLAP_AUDIO_DIR, f"{s}.npy"))]
    print(f"Audio: {len(sample_ids)-len(to_extract)} cached, {len(to_extract)} to extract")

    for sid in tqdm(to_extract, desc="CLAP audio"):
        audio_path = os.path.join(EVAL_AUDIO_DIR, f"{sid}.wav")
        try:
            audio_data, sr = sf.read(audio_path, dtype='float32')
            if audio_data.ndim == 2:
                audio_data = audio_data.mean(axis=1)
            if sr != 48000:
                audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)
            with torch.no_grad():
                embed = clap_module.get_audio_embedding_from_data(x=[audio_data.astype(np.float32)], use_tensor=False)
            np.save(os.path.join(EVAL_CLAP_AUDIO_DIR, f"{sid}.npy"), embed[0])
        except Exception as e:
            print(f"  Error {sid}: {e}")

    # Text embeddings
    os.makedirs(EVAL_CLAP_TEXT_DIR, exist_ok=True)
    to_extract_text = [(row['anonymous_id'], row) for _, row in eval_df.iterrows()
                       if not os.path.exists(os.path.join(EVAL_CLAP_TEXT_DIR, f"{row['anonymous_id']}.npy"))]
    print(f"Text: {len(sample_ids)-len(to_extract_text)} cached, {len(to_extract_text)} to extract")

    for i in tqdm(range(0, len(to_extract_text), 64), desc="CLAP text"):
        batch = to_extract_text[i:i+64]
        texts = [clean_text(r.get('title',''), r.get('tags',''), r.get('description','')) for _, r in batch]
        sids = [s for s, _ in batch]
        with torch.no_grad():
            embeds = clap_module.get_text_embedding(texts, use_tensor=False)
        for sid, emb in zip(sids, embeds):
            np.save(os.path.join(EVAL_CLAP_TEXT_DIR, f"{sid}.npy"), emb)

    del clap_module
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("✅ CLAP embeddings done")


def extract_convnext_embeddings(device):
    """Extract ConvNeXt audio embeddings for eval set."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'convNeXt_model'))
    from audioset_convnext_inf.pytorch.convnext import ConvNeXt

    eval_df = pd.read_csv(EVAL_METADATA)
    sample_ids = eval_df['anonymous_id'].tolist()
    print(f"\n{'='*60}\nCONVNEXT EMBEDDING EXTRACTION ({len(sample_ids)} samples)\n{'='*60}")

    CHECKPOINT = "convNeXt_model/convnext_tiny_465mAP_BL_AC_70kit.pth"
    model = ConvNeXt(in_chans=1, num_classes=527, use_torchaudio=False)
    model.downsample_layers[0][0] = torch.nn.Conv2d(1, 96, kernel_size=(4,4), stride=(4,4))
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    os.makedirs(EVAL_CONVNEXT_DIR, exist_ok=True)
    to_extract = [s for s in sample_ids if not os.path.exists(os.path.join(EVAL_CONVNEXT_DIR, f"{s}.npy"))]
    print(f"ConvNeXt: {len(sample_ids)-len(to_extract)} cached, {len(to_extract)} to extract")

    SR = 32000
    for sid in tqdm(to_extract, desc="ConvNeXt"):
        audio_path = os.path.join(EVAL_AUDIO_DIR, f"{sid}.wav")
        try:
            data, sr = sf.read(audio_path, dtype='float32')
            waveform = torch.from_numpy(data).unsqueeze(0) if data.ndim == 1 else torch.from_numpy(data.T)
            if sr != SR:
                import torchaudio
                waveform = torchaudio.functional.resample(waveform, sr, SR)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if waveform.shape[1] < SR:
                waveform = torch.nn.functional.pad(waveform, (0, SR - waveform.shape[1]))
            with torch.no_grad():
                frame_emb = model.forward_frame_embeddings(waveform.to(device))
                scene_emb = frame_emb.mean(dim=[2,3]).squeeze(0).cpu().numpy()
            np.save(os.path.join(EVAL_CONVNEXT_DIR, f"{sid}.npy"), scene_emb)
        except Exception as e:
            print(f"  Error {sid}: {e}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("✅ ConvNeXt embeddings done")


def extract_whisper_embeddings(device):
    """Extract Whisper audio embeddings for eval set."""
    from transformers import WhisperModel, WhisperFeatureExtractor

    eval_df = pd.read_csv(EVAL_METADATA)
    sample_ids = eval_df['anonymous_id'].tolist()
    print(f"\n{'='*60}\nWHISPER EMBEDDING EXTRACTION ({len(sample_ids)} samples)\n{'='*60}")

    MODEL_NAME = "openai/whisper-base"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
    model = WhisperModel.from_pretrained(MODEL_NAME).encoder.to(device)
    model.eval()

    os.makedirs(EVAL_WHISPER_DIR, exist_ok=True)
    to_extract = [s for s in sample_ids if not os.path.exists(os.path.join(EVAL_WHISPER_DIR, f"{s}.npy"))]
    print(f"Whisper: {len(sample_ids)-len(to_extract)} cached, {len(to_extract)} to extract")

    SR = 16000
    for sid in tqdm(to_extract, desc="Whisper"):
        try:
            audio, sr = sf.read(os.path.join(EVAL_AUDIO_DIR, f"{sid}.wav"), dtype='float32')
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SR:
                ratio = SR / sr
                new_len = int(len(audio) * ratio)
                audio = np.interp(np.linspace(0, len(audio)-1, new_len), np.arange(len(audio)), audio).astype(np.float32)
            if len(audio) > 30 * SR:
                audio = audio[:30*SR]
            inputs = feature_extractor(audio, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                enc_out = model(inputs.input_features.to(device)).last_hidden_state
                embedding = enc_out.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
            np.save(os.path.join(EVAL_WHISPER_DIR, f"{sid}.npy"), embedding)
        except Exception as e:
            print(f"  Error {sid}: {e}")

    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("✅ Whisper embeddings done")


def extract_clap_ft_logits(device):
    """Run fine-tuned CLAP end-to-end on eval set → logits per fold."""
    from huggingface_hub import hf_hub_download
    import laion_clap
    from finetune_clap import CLAPFinetuneModel, CLAPAudioPreprocessor
    from models import BaseClassifier

    eval_df = pd.read_csv(EVAL_METADATA)
    sample_ids = eval_df['anonymous_id'].tolist()

    output_dir = "data/eval/features/clap_ft_logits"
    if os.path.exists(os.path.join(output_dir, "fold_4_logits.npy")):
        print("✅ clap_ft logits already cached")
        return
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}\nFINE-TUNED CLAP INFERENCE ({len(sample_ids)} samples)\n{'='*60}")

    # Load base CLAP
    clap_module = laion_clap.CLAP_Module(enable_fusion=True, amodel="HTSAT-tiny")
    clap_ckpt = hf_hub_download(repo_id="lukewys/laion_clap", filename="630k-audioset-fusion-best.pt")
    clap_module.load_ckpt(clap_ckpt)

    preprocessor = CLAPAudioPreprocessor()

    for fold in range(NUM_FOLDS):
        out_path = os.path.join(output_dir, f"fold_{fold}_logits.npy")
        ids_path = os.path.join(output_dir, f"fold_{fold}_ids.npy")
        if os.path.exists(out_path):
            print(f"  Fold {fold}: cached")
            continue

        ckpt_path = f"model_output_finetune/both/fold_{fold}/best_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"  Fold {fold}: checkpoint not found!")
            continue

        hatr = BaseClassifier(hidden_size=128, num_classes=23,
                              emb_size_audio=512, emb_size_text=512,
                              dropout=0.1, use_batch_norm=True, mode='both')
        model = CLAPFinetuneModel(clap_module, hatr, freeze_early_layers=True)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.clap_model.audio_branch.load_state_dict(ckpt['clap_audio_branch'])
        model.clap_model.audio_projection.load_state_dict(ckpt['clap_audio_projection'])
        model.hatr.load_state_dict(ckpt['hatr'])
        model.to(device).eval()

        all_logits = []
        MAX_SAMPLES = 48000 * 10  # 10s at 48kHz
        for sid in tqdm(sample_ids, desc=f"clap_ft fold {fold}"):
            audio_path = os.path.join(EVAL_AUDIO_DIR, f"{sid}.wav")
            text_emb = np.load(os.path.join(EVAL_CLAP_TEXT_DIR, f"{sid}.npy"))
            text_emb_t = torch.tensor(text_emb, dtype=torch.float32).unsqueeze(0).to(device)

            try:
                audio_data, sr = sf.read(audio_path, dtype='float32')
                if audio_data.ndim == 2:
                    audio_data = audio_data.mean(axis=1)
                if sr != 48000:
                    audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=48000)
                # Pad or truncate to 10s
                if len(audio_data) > MAX_SAMPLES:
                    audio_data = audio_data[:MAX_SAMPLES]
                else:
                    audio_data = np.pad(audio_data, (0, MAX_SAMPLES - len(audio_data)))
                waveform = torch.tensor(audio_data, dtype=torch.float32)

                # Build mel_fusion: (4, time, mel_bins)
                fbank = preprocessor(waveform)  # (target_length, mel_bins)
                mel_fusion = fbank.unsqueeze(0).repeat(4, 1, 1)
                mel_fusion[1:, :, :] = 0.0  # zero channels 1-3

                mel_fusion = mel_fusion.unsqueeze(0).to(device)
                waveform = waveform.unsqueeze(0).to(device)

                with torch.no_grad():
                    _, class_logits, _, _, _ = model(mel_fusion, waveform, text_emb_t)
                all_logits.append(class_logits.cpu().numpy())
            except Exception as e:
                all_logits.append(np.zeros((1, 23), dtype=np.float32))
                print(f"    Error {sid}: {e}")

        logits_arr = np.concatenate(all_logits, axis=0)
        np.save(out_path, logits_arr)
        np.save(ids_path, np.array(sample_ids))
        print(f"  Fold {fold}: ✅ {logits_arr.shape}")

        del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    print("✅ clap_ft logits done")


# ═══════════════════════════════════════════════════════════════
# STEP 2: INFERENCE
# ═══════════════════════════════════════════════════════════════

class EvalDataset(Dataset):
    def __init__(self, sample_ids, audio_emb_dir, text_emb_dir, aux_emb_dir=None):
        self.sample_ids = sample_ids
        self.audio_emb_dir = audio_emb_dir
        self.text_emb_dir = text_emb_dir
        self.aux_emb_dir = aux_emb_dir

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        sid = self.sample_ids[idx]
        item = {
            'sound_id': sid,
            'audio_embedding': torch.tensor(np.load(os.path.join(self.audio_emb_dir, f"{sid}.npy")), dtype=torch.float32),
            'text_embedding': torch.tensor(np.load(os.path.join(self.text_emb_dir, f"{sid}.npy")), dtype=torch.float32),
        }
        if self.aux_emb_dir:
            item['aux_embedding'] = torch.tensor(np.load(os.path.join(self.aux_emb_dir, f"{sid}.npy")), dtype=torch.float32)
        return item


def load_hatr_model(model_dir, fold, device):
    model_path = os.path.join(model_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(model_path):
        return None
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = BaseClassifier(**config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    return model


AUDIO_DIR_MAP = {
    "clap": EVAL_CLAP_AUDIO_DIR,
    "convnext": EVAL_CONVNEXT_DIR,
}
AUX_DIR_MAP = {
    "whisper": EVAL_WHISPER_DIR,
}


def run_submission_inference(sub_id, sub_config, device):
    """Run inference for one submission config."""
    eval_df = pd.read_csv(EVAL_METADATA)
    sample_ids = eval_df['anonymous_id'].tolist()

    print(f"\n{'='*60}")
    print(f"SUBMISSION {sub_id}: {sub_config['name']}")
    print(f"{'='*60}")

    all_logits = []

    # Standard HATR models
    for model_name, model_dir, audio_type, aux_type in sub_config["models"]:
        print(f"\n  Model: {model_name} (audio={audio_type}, aux={aux_type})")
        audio_dir = AUDIO_DIR_MAP[audio_type]
        aux_dir = AUX_DIR_MAP.get(aux_type) if aux_type else None

        dataset = EvalDataset(sample_ids, audio_dir, EVAL_CLAP_TEXT_DIR, aux_dir)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

        fold_logits = []
        for fold in range(NUM_FOLDS):
            model = load_hatr_model(model_dir, fold, device)
            if model is None:
                print(f"    Fold {fold}: MISSING")
                continue

            logits_list = []
            with torch.no_grad():
                for batch in loader:
                    audio_emb = batch['audio_embedding'].to(device)
                    text_emb = batch['text_embedding'].to(device)
                    aux_emb = batch.get('aux_embedding')
                    if aux_emb is not None:
                        aux_emb = aux_emb.to(device)
                    _, class_logits, _, _ = model(audio_emb, text_emb, aux_emb)
                    logits_list.append(class_logits.cpu().numpy())

            fold_logits.append(np.concatenate(logits_list, axis=0))
            print(f"    Fold {fold}: ✅")
            del model

        if fold_logits:
            all_logits.append(np.mean(fold_logits, axis=0))
            print(f"    → Averaged {len(fold_logits)} folds")

    # Add clap_ft logits if needed
    if sub_config.get("clap_ft"):
        print(f"\n  Model: clap_ft (end-to-end fine-tuned CLAP)")
        ft_dir = "data/eval/features/clap_ft_logits"
        fold_logits = []
        for fold in range(NUM_FOLDS):
            lpath = os.path.join(ft_dir, f"fold_{fold}_logits.npy")
            if os.path.exists(lpath):
                fold_logits.append(np.load(lpath))
                print(f"    Fold {fold}: ✅ (precomputed)")
            else:
                print(f"    Fold {fold}: MISSING")
        if fold_logits:
            all_logits.append(np.mean(fold_logits, axis=0))
            print(f"    → Averaged {len(fold_logits)} folds")

    # Ensemble average
    ensemble_logits = np.mean(all_logits, axis=0)
    predictions = ensemble_logits.argmax(axis=1)

    # Softmax for confidence scores
    from scipy.special import softmax
    probs = softmax(ensemble_logits, axis=1)
    confidence = probs.max(axis=1)

    print(f"  ✅ Predictions: {len(predictions)} samples")
    return sample_ids, predictions, confidence


# ═══════════════════════════════════════════════════════════════
# STEP 3: OUTPUT CSV
# ═══════════════════════════════════════════════════════════════

def generate_output_csv(sample_ids, predictions, confidence, sub_id):
    """Generate DCASE output CSV: id,predicted_bst_second_level_class,prediction_score"""
    class_dict = json.load(open(CLASS_DICT_PATH))
    idx_to_class = {v: k for k, v in class_dict.items()}

    pred_classes = [idx_to_class[int(p)] for p in predictions]

    output_path = f"{TEAM}_task1_{sub_id}.output.csv"
    df = pd.DataFrame({
        'id': sample_ids,
        'predicted_bst_second_level_class': pred_classes,
        'prediction_score': [f"{c:.4f}" for c in confidence],
    })
    df.to_csv(output_path, index=False)
    print(f"\n  Saved: {output_path} ({len(df)} rows)")

    # Distribution
    for cls, cnt in df['predicted_bst_second_level_class'].value_counts().head(5).items():
        print(f"    {cls}: {cnt} ({100*cnt/len(df):.1f}%)")
    return output_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="DCASE 2026 Task 1 Eval Inference - All Submissions")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip embedding extraction")
    parser.add_argument("--sub", type=int, default=0, help="Run only this submission (1-4), 0=all")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Determine which submissions to run
    subs_to_run = [args.sub] if args.sub > 0 else [1, 2, 3, 4]

    # Determine which embeddings we need
    need_clap = any(s in subs_to_run for s in [1, 2, 3, 4])
    need_convnext = any(s in subs_to_run for s in [1, 2])
    need_whisper = 4 in subs_to_run
    need_clap_ft = 1 in subs_to_run

    # Step 1: Extract embeddings
    if not args.skip_extraction:
        if need_clap:
            extract_clap_embeddings(device)
        if need_convnext:
            extract_convnext_embeddings(device)
        if need_whisper:
            extract_whisper_embeddings(device)
        if need_clap_ft:
            extract_clap_ft_logits(device)
    else:
        print("Skipping embedding extraction (using cached)")

    # Step 2-3: Inference + CSV for each submission
    outputs = []
    for sub_id in subs_to_run:
        result = run_submission_inference(sub_id, SUBMISSIONS[sub_id], device)
        if result:
            sample_ids, predictions, confidence = result
            out = generate_output_csv(sample_ids, predictions, confidence, sub_id)
            outputs.append(out)

    print(f"\n{'='*60}")
    print(f"ALL DONE! Generated {len(outputs)} output files:")
    for f in outputs:
        print(f"  → {f}")
    print(f"{'='*60}")
    print(f"\nNext: bash assemble_submission.sh")


if __name__ == "__main__":
    main()
