"""Per-class analysis: compute F1, precision, recall for each class across all folds."""
import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict

def analyze_model(model_dir, mode="both"):
    """Aggregate predictions across folds and compute per-class metrics."""
    all_preds = []
    for fold in range(5):
        pred_path = os.path.join(model_dir, mode, f"fold_{fold}", "evaluation", "predictions.csv")
        if os.path.exists(pred_path):
            df = pd.read_csv(pred_path)
            df['fold'] = fold
            all_preds.append(df)
    
    if not all_preds:
        print(f"No predictions found in {model_dir}/{mode}/")
        return None
    
    all_df = pd.concat(all_preds, ignore_index=True)
    
    classes = sorted(all_df['ground_truth'].unique())
    
    results = []
    for cls in classes:
        tp = ((all_df['ground_truth'] == cls) & (all_df['prediction'] == cls)).sum()
        fp = ((all_df['ground_truth'] != cls) & (all_df['prediction'] == cls)).sum()
        fn = ((all_df['ground_truth'] == cls) & (all_df['prediction'] != cls)).sum()
        support = (all_df['ground_truth'] == cls).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Top-class accuracy (correct top-level even if subclass wrong)
        top_cls = cls.split('-')[0]
        mask_gt = all_df['ground_truth'] == cls
        if mask_gt.sum() > 0:
            pred_tops = all_df.loc[mask_gt, 'prediction'].apply(lambda x: x.split('-')[0])
            top_acc = (pred_tops == top_cls).mean()
        else:
            top_acc = 0
        
        # Most common confusions
        confused_with = all_df.loc[(all_df['ground_truth'] == cls) & (all_df['prediction'] != cls), 'prediction']
        top_confusions = confused_with.value_counts().head(3)
        conf_str = ", ".join([f"{c}({n})" for c, n in top_confusions.items()])
        
        results.append({
            'class': cls,
            'top_class': top_cls,
            'support': support,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'top_acc': top_acc,
            'errors': fn,
            'top_confusions': conf_str,
        })
    
    df_results = pd.DataFrame(results)
    return df_results

if __name__ == "__main__":
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "model_output_cw_only"
    mode = sys.argv[2] if len(sys.argv) > 2 else "both"
    
    print(f"\n{'='*80}")
    print(f"Per-class analysis: {model_dir} ({mode} mode)")
    print(f"{'='*80}\n")
    
    df = analyze_model(model_dir, mode)
    if df is None:
        sys.exit(1)
    
    # Sort by F1 ascending (worst first)
    df_sorted = df.sort_values('f1')
    
    total = df['support'].sum()
    correct = df.apply(lambda r: int(r['recall'] * r['support']), axis=1).sum()
    
    print(f"Total samples: {total}, Overall accuracy: {100*correct/total:.1f}%")
    print(f"Classes: {len(df)}, Macro F1: {100*df['f1'].mean():.2f}%\n")
    
    # Worst classes
    print("=== WORST 15 CLASSES (by F1) ===")
    print(f"{'Class':<8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TopAcc':>7} {'Supp':>5} {'Err':>4}  Top Confusions")
    print("-" * 90)
    for _, r in df_sorted.head(15).iterrows():
        print(f"{r['class']:<8} {100*r['f1']:>5.1f}% {100*r['precision']:>5.1f}% {100*r['recall']:>5.1f}% {100*r['top_acc']:>6.1f}% {r['support']:>5} {r['errors']:>4}  {r['top_confusions']}")
    
    # Best classes
    print(f"\n=== BEST 10 CLASSES (by F1) ===")
    print(f"{'Class':<8} {'F1':>6} {'Prec':>6} {'Rec':>6} {'TopAcc':>7} {'Supp':>5}")
    print("-" * 50)
    for _, r in df_sorted.tail(10).iterrows():
        print(f"{r['class']:<8} {100*r['f1']:>5.1f}% {100*r['precision']:>5.1f}% {100*r['recall']:>5.1f}% {100*r['top_acc']:>6.1f}% {r['support']:>5}")
    
    # Per top-class summary
    print(f"\n=== TOP-CLASS SUMMARY ===")
    top_summary = df.groupby('top_class').agg(
        f1_mean=('f1', 'mean'),
        f1_min=('f1', 'min'),
        support=('support', 'sum'),
        num_subclasses=('class', 'count'),
    ).sort_values('f1_mean')
    
    print(f"{'TopClass':<8} {'AvgF1':>6} {'MinF1':>6} {'Supp':>6} {'#Sub':>4}")
    print("-" * 35)
    for tc, r in top_summary.iterrows():
        print(f"{tc:<8} {100*r['f1_mean']:>5.1f}% {100*r['f1_min']:>5.1f}% {r['support']:>6} {int(r['num_subclasses']):>4}")
    
    # Save full results
    out_path = os.path.join(model_dir, f"per_class_analysis_{mode}.csv")
    df_sorted.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")
