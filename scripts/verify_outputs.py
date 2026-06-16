import os, json, sys
import numpy as np
import pandas as pd

EVAL_METADATA = "data/eval/metadata.csv"
CLASS_DICT_PATH = "data/class_dict.json"
TEAM = "Kucukoglu_NYU"
NUM_CLASSES = 23
UNIFORM_CONF = 1.0 / NUM_CLASSES  # ~0.0435 -> signature of a zero-logit fallback row

def main():
    eval_df = pd.read_csv(EVAL_METADATA)
    expected_ids = eval_df["anonymous_id"].tolist()
    expected_set = set(expected_ids)
    n_expected = len(expected_ids)

    class_dict = json.load(open(CLASS_DICT_PATH))
    valid_classes = set(class_dict.keys())
    if len(class_dict) != NUM_CLASSES:
        print(f"⚠️  class_dict has {len(class_dict)} classes but NUM_CLASSES={NUM_CLASSES} "
              f"-> idx_to_class mapping may be wrong")

    print(f"Expected {n_expected} samples, {len(valid_classes)} classes\n")

    all_preds = {}
    ok = True
    for sub in (1, 2, 3, 4):
        path = f"results/{TEAM}_task1_{sub}.output.csv"
        if not os.path.exists(path):
            print(f"❌ sub {sub}: file missing ({path})"); ok = False; continue

        df = pd.read_csv(path)
        ids = df["id"].tolist()
        problems = []

        # completeness
        if len(df) != n_expected:
            problems.append(f"row count {len(df)} != {n_expected}")
        if df["id"].duplicated().any():
            problems.append(f"{df['id'].duplicated().sum()} duplicate ids")
        missing = expected_set - set(ids)
        extra = set(ids) - expected_set
        if missing: problems.append(f"{len(missing)} missing ids")
        if extra:   problems.append(f"{len(extra)} unexpected ids")

        # validity
        bad_cls = set(df["predicted_bst_second_level_class"]) - valid_classes
        if bad_cls: problems.append(f"unknown classes: {bad_cls}")
        scores = pd.to_numeric(df["prediction_score"], errors="coerce")
        if scores.isna().any():
            problems.append(f"{scores.isna().sum()} unparseable scores")
        elif scores.min() < 0 or scores.max() > 1.0001:
            problems.append(f"scores out of [0,1]: [{scores.min():.3f},{scores.max():.3f}]")

        # collapsed distribution
        vc = df["predicted_bst_second_level_class"].value_counts()
        top_frac = vc.iloc[0] / len(df)
        if top_frac > 0.5:
            problems.append(f"distribution collapsed: '{vc.index[0]}' = {top_frac:.0%}")
        n_classes_used = df["predicted_bst_second_level_class"].nunique()

        # zero-logit fallback signature (clap_ft) -> uniform-confidence class-0 rows
        near_uniform = (scores - UNIFORM_CONF).abs() < 1e-3
        n_fallback = int(near_uniform.sum())
        if n_fallback:
            problems.append(f"{n_fallback} rows at uniform conf ~{UNIFORM_CONF:.3f} "
                            f"(likely zero-logit fallback)")

        all_preds[sub] = df.set_index("id")["predicted_bst_second_level_class"]

        if problems:
            ok = False
            print(f"❌ sub {sub}: " + "; ".join(problems))
        else:
            print(f"✅ sub {sub}: {len(df)} rows, {n_classes_used}/{NUM_CLASSES} classes used, "
                  f"top class {top_frac:.0%}, conf {scores.mean():.3f} mean")

    # cross-check: different configs should not produce identical predictions
    print()
    pairs = [(1,2),(1,3),(2,4),(3,4)]
    for a,b in pairs:
        if a in all_preds and b in all_preds:
            common = all_preds[a].index.intersection(all_preds[b].index)
            agree = (all_preds[a].loc[common] == all_preds[b].loc[common]).mean()
            note = "  ⚠️ suspiciously identical" if agree > 0.999 else ""
            print(f"sub {a} vs {b}: {agree:.1%} agreement{note}")

    print("\n" + ("✅ ALL CHECKS PASSED" if ok else "❌ ISSUES FOUND — see above"))
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
