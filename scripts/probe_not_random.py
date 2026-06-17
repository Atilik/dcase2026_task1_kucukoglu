"""Probe: are the sub-3 (xswap) predictions from the TRAINED model, or random?

Place in scripts/ and run from the repo root:
    python scripts/probe_not_random.py

CPU-only, uses cached eval CLAP embeddings, needs NO labels.
Proves:
  1. The checkpoint loads with a full key match (no silent partial load).
  2. The trained model is decisive where a random-init model is not.
  3. Trained predictions are unrelated to random init (agreement ~= chance),
     but two independently-trained folds AGREE with each other a lot
     (only happens if they learned real structure from data).
  4. Inference is deterministic (no randomness leaking in at predict time).
"""
import os, json
import numpy as np
import pandas as pd
import torch
from models import BaseClassifier

EVAL_METADATA = "data/eval/metadata.csv"
CLAP_AUDIO = "data/eval/features/clap_audio_embeddings"
CLAP_TEXT  = "data/eval/features/clap_text_embeddings"
MODEL_DIR  = "model_outputs/model_output_xswap_noise/both"
N = 300  # subset of eval samples is plenty to see the signal

ids = pd.read_csv(EVAL_METADATA)["anonymous_id"].tolist()[:N]
A = torch.tensor(np.stack([np.load(f"{CLAP_AUDIO}/{s}.npy") for s in ids]), dtype=torch.float32)
T = torch.tensor(np.stack([np.load(f"{CLAP_TEXT}/{s}.npy")  for s in ids]), dtype=torch.float32)
print(f"Loaded {len(ids)} eval samples | audio {tuple(A.shape)} | text {tuple(T.shape)}")


def logits_from(model):
    with torch.no_grad():
        out = model(A, T, None)
    cl = out[1] if isinstance(out, (tuple, list)) else out
    return cl.cpu().numpy()


def load_trained(fold):
    ck = torch.load(f"{MODEL_DIR}/fold_{fold}/best_model.pth",
                    map_location="cpu", weights_only=False)
    m = BaseClassifier(**ck["config"])
    # strict=False here only so we can REPORT mismatches instead of crashing
    res = m.load_state_dict(ck["model_state"], strict=False)
    m.eval()
    return m, ck["config"], res.missing_keys, res.unexpected_keys


# 1. Load trained fold 0 + verify full key match
m0, cfg, missing, unexpected = load_trained(0)
ncls = cfg.get("num_classes", 23)
print(f"\n[1] Checkpoint key check (fold 0): "
      f"{len(missing)} missing, {len(unexpected)} unexpected"
      + ("  -> FULL MATCH" if not missing and not unexpected else "  -> PARTIAL LOAD!"))

L_trained = logits_from(m0)
pred_trained = L_trained.argmax(1)
conf_trained = torch.softmax(torch.tensor(L_trained), 1).max(1).values.numpy()

# 2. Same architecture, random init
torch.manual_seed(0)
m_rand = BaseClassifier(**cfg).eval()
L_rand = logits_from(m_rand)
pred_rand = L_rand.argmax(1)
conf_rand = torch.softmax(torch.tensor(L_rand), 1).max(1).values.numpy()

# 3. A second independently-trained fold
m1, *_ = load_trained(1)
pred_fold1 = logits_from(m1).argmax(1)

# 4. Re-run fold 0 for determinism
L_again = logits_from(m0)

chance = 1.0 / ncls
print(f"\n[2] Mean top-class confidence (decisiveness):")
print(f"      trained fold0 : {conf_trained.mean():.3f}")
print(f"      random init   : {conf_rand.mean():.3f}   (should be much lower / fuzzier)")

print(f"\n[3] Prediction agreement (fraction identical class):")
print(f"      trained0 vs random   : {(pred_trained==pred_rand).mean():.3f}   "
      f"(chance ~{chance:.3f} -> trained preds owe nothing to random init)")
print(f"      trained0 vs trained1 : {(pred_trained==pred_fold1).mean():.3f}   "
      f"(two real folds should agree FAR above chance)")

print(f"\n[4] Determinism: max logit diff on identical re-run = "
      f"{np.abs(L_trained - L_again).max():.2e}   (should be ~0)")

print("\nVerdict:")
ok = (not missing and not unexpected
      and conf_trained.mean() > 2*conf_rand.mean()
      and (pred_trained==pred_fold1).mean() > 0.4
      and (pred_trained==pred_rand).mean() < 0.15)
print("  ✅ Predictions come from the trained weights, deterministically."
      if ok else
      "  ⚠️ Something looks off — inspect the numbers above.")
