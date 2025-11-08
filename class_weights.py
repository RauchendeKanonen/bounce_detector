#!/usr/bin/env python3
import argparse, numpy as np, os, sys, json
from glob import glob

def summarize_file(path):
    npz = np.load(path, allow_pickle=True)
    labels = npz["labels"].astype(np.int8)
    valid_idx = npz["valid_idx"].astype(np.int64)

    # Labels used for training/prediction are exactly at valid_idx
    y_used = labels[valid_idx]

    # Safety: keep only 0/1; ignore anything else (e.g., -1 if any slipped in)
    y_used = y_used[(y_used == 0) | (y_used == 1)]

    pos = int((y_used == 1).sum())
    neg = int((y_used == 0).sum())
    return pos, neg, len(valid_idx), len(labels)

def main():
    ap = argparse.ArgumentParser(description="Compute global class weight from labeled NPZ files (using labels at valid_idx).")
    ap.add_argument("inputs", nargs="+", help="NPZ paths or globs (e.g., data/*_labeled_events_*.npz)")
    ap.add_argument("--json", dest="out_json", help="Optional path to write a JSON summary.")
    args = ap.parse_args()

    # Expand globs
    paths = []
    for p in args.inputs:
        paths.extend(glob(p))
    paths = sorted(set(paths))

    if not paths:
        print("No files matched.")
        sys.exit(2)

    total_pos = total_neg = 0
    rows_total = vi_total = 0
    per_file = []

    for p in paths:
        try:
            pos, neg, vi, rows = summarize_file(p)
        except Exception as e:
            print(f"[WARN] Skipping {p}: {e}")
            continue

        total_pos += pos
        total_neg += neg
        rows_total += rows
        vi_total += vi

        rate = (pos / max(pos + neg, 1)) if (pos + neg) else 0.0
        per_file.append({
            "file": os.path.basename(p),
            "valid_idx": vi,
            "labels_used": pos + neg,
            "pos": pos,
            "neg": neg,
            "pos_rate": round(rate, 6),
            "note": "labels_used counts only 0/1 at valid_idx"
        })

    # Overall
    used_total = total_pos + total_neg
    pos_rate = (total_pos / used_total) if used_total else 0.0
    if total_pos == 0:
        pos_weight = float("inf")  # can't divide by zero
        weight_note = "No positives found; consider smoothing or revisiting labeling."
    else:
        pos_weight = total_neg / total_pos
        weight_note = ""

    print("\n=== Class balance over all files (at valid_idx) ===")
    print(f" files             : {len(per_file)}")
    print(f" rows total        : {rows_total}")
    print(f" valid_idx total   : {vi_total}")
    print(f" labels used (0/1) : {used_total}")
    print(f" positives         : {total_pos}")
    print(f" negatives         : {total_neg}")
    print(f" positive rate     : {pos_rate:.8f}")
    print(f" pos_weight        : {pos_weight if np.isfinite(pos_weight) else 'inf'}")
    if weight_note:
        print(f" note              : {weight_note}")

    print("\nPer-file summary:")
    for row in per_file[:20]:  # print first 20 to keep it tidy
        print(f" - {row['file']}: valid_idx={row['valid_idx']}, used={row['labels_used']}, "
              f"+={row['pos']}, 0={row['neg']}, pos_rate={row['pos_rate']}")

    # Optional JSON dump
    if args.out_json:
        out = {
            "files": per_file,
            "totals": {
                "rows_total": rows_total,
                "valid_idx_total": vi_total,
                "labels_used_total": used_total,
                "pos_total": total_pos,
                "neg_total": total_neg,
                "pos_rate": pos_rate,
                "pos_weight": None if not np.isfinite(pos_weight) else pos_weight,
            }
        }
        with open(args.out_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote JSON summary to {args.out_json}")

if __name__ == "__main__":
    main()
