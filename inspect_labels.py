#!/usr/bin/env python3
import sys
import numpy as np
import os

def inspect_file(path):
    npz = np.load(path, allow_pickle=True)
    labels = npz["labels"]
    valid_idx = npz["valid_idx"]
    mid = npz["sampled"]["mid"]

    # Basic counts
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    n_invalid = int((labels == -1).sum())
    print(f"\n{os.path.basename(path)}")
    print(f"  total samples: {len(labels)}")
    print(f"  valid_idx: {len(valid_idx)} | +1={n_pos}, 0={n_neg}, -1={n_invalid}")

    # Sanity: show first few valid_idx with mid price and label
    print("  First 10 valid_idx entries:")
    for j, idx in enumerate(valid_idx[:10]):
        print(f"    {j:2d}: idx={idx:6d}, mid={mid[idx]:.6f}, label={labels[idx]}")

    # Optional: check gain distribution if stored
    if "events" in npz:
        gains = npz["events"]["gain"]
        if len(gains):
            print(f"  gains: min={gains.min():.5f}, mean={gains.mean():.5f}, max={gains.max():.5f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_labels.py <labeled1.npz> [labeled2.npz ...]")
        sys.exit(1)
    for f in sys.argv[1:]:
        inspect_file(f)
