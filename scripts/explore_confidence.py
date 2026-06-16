"""Explore the confidence column in BSD10k-v1.2 metadata."""
import pandas as pd
import numpy as np

df = pd.read_csv("data/BSD10k-v1.2/metadata/BSD10k_metadata.csv")

print(f"Total samples: {len(df)}")
print(f"\n=== Confidence Stats ===")
print(f"Min:    {df['confidence'].min():.4f}")
print(f"Max:    {df['confidence'].max():.4f}")
print(f"Mean:   {df['confidence'].mean():.4f}")
print(f"Median: {df['confidence'].median():.4f}")
print(f"Std:    {df['confidence'].std():.4f}")

print(f"\n=== Value Distribution ===")
vc = df['confidence'].value_counts().sort_index()
for val, count in vc.items():
    print(f"  {val:.2f}: {count:5d} ({100*count/len(df):.1f}%)")

thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\n=== Samples Remaining at Thresholds ===")
for t in thresholds:
    remaining = df[df['confidence'] >= t]
    print(f"  >= {t:.1f}: {len(remaining):5d} samples ({100*len(remaining)/len(df):.1f}%)")

print(f"\n=== Class Distribution at Key Thresholds ===")
for t in [0.0, 0.7, 0.8, 0.9]:
    subset = df[df['confidence'] >= t]
    print(f"\n  --- Threshold >= {t} ({len(subset)} samples) ---")
    class_counts = subset['class'].value_counts().sort_values()
    for cls, count in class_counts.items():
        orig_count = len(df[df['class'] == cls])
        print(f"    {cls:30s}: {count:4d} / {orig_count:4d} ({100*count/orig_count:.0f}%)")
