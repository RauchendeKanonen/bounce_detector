#!/usr/bin/env python3
import argparse
import math
import os
import random
import re
import shutil
from pathlib import Path
from typing import List, Dict, Sequence, Tuple

PATTERN = re.compile(r'^([A-Za-z0-9.\-]+).*?(Lo|Hi)\.npz$', re.IGNORECASE)

def list_matching_files(src: Path, recursive: bool) -> List[Path]:
    globber = src.rglob if recursive else src.glob
    return [p for p in globber("*.npz") if PATTERN.search(p.name)]

def group_by_symbol(files: List[Path]) -> Dict[str, List[Path]]:
    groups: Dict[str, List[Path]] = {}
    for f in files:
        m = PATTERN.search(f.name)
        if not m:
            continue
        sym = m.group(1)
        groups.setdefault(sym, []).append(f)
    return groups

def compute_quota(n: int, pct: Sequence[int]) -> Tuple[int, int, int]:
    """Largest-remainder method to hit targets exactly (up to rounding)."""
    raw = [n * p / 100.0 for p in pct]
    floors = [math.floor(x) for x in raw]
    remaining = n - sum(floors)
    # Assign remaining ones to the largest fractional parts
    fracs = sorted(enumerate([r - f for r, f in zip(raw, floors)]),
                   key=lambda t: t[1], reverse=True)
    for i in range(remaining):
        floors[fracs[i][0]] += 1
    return tuple(floors)  # (q0, q1, q2)

def quota_split(units: List, pct: Sequence[int]) -> List[List]:
    """Shuffle then split units into three lists meeting the quota exactly."""
    random.shuffle(units)
    q0, q1, q2 = compute_quota(len(units), pct)
    return [units[:q0], units[q0:q0+q1], units[q0+q1:]]  # q2 implied

def move_batch(items: List[List[Path]], out_dirs: List[Path], commit: bool):
    for bucket_idx, group in enumerate(items):
        for f in group:
            dest = out_dirs[bucket_idx] / f.name
            if not commit:
                print(f"[DRY-RUN] would move: {f} -> {dest}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            final_dest = dest
            counter = 1
            while final_dest.exists():
                final_dest = dest.with_name(f"{dest.stem}__{counter}{dest.suffix}")
                counter += 1
            shutil.move(str(f), str(final_dest))
            print(f"Moved: {f} -> {final_dest}")

def main():
    ap = argparse.ArgumentParser(
        description="Move *.npz files ending in Lo.npz or Hi.npz into three subdirectories by percentage."
    )
    ap.add_argument("--src", type=Path, required=True, help="Source directory to scan")
    ap.add_argument("--out", nargs=3, metavar=("OUT1", "OUT2", "OUT3"), required=True,
                    help="Names of three subdirectories to create inside --src (or absolute paths)")
    ap.add_argument("--pct", nargs=3, type=int, metavar=("P1", "P2", "P3"), default=[70, 15, 15],
                    help="Percentages for the three buckets (must sum to 100)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    ap.add_argument("--by-symbol", action="store_true",
                    help="Distribute by symbol (keep all files of a ticker together)")
    ap.add_argument("--recursive", action="store_true", help="Recurse into subdirectories")
    ap.add_argument("--random", action="store_true",
                    help="Randomize order but enforce exact quotas (vs deterministic slicing)")
    ap.add_argument("--commit", action="store_true",
                    help="Actually move files. Without this flag, performs a dry run.")
    args = ap.parse_args()

    if sum(args.pct) != 100 or any(p < 0 for p in args.pct):
        ap.error("--pct must be three non-negative integers that sum to 100")

    random.seed(args.seed)

    # Resolve output dirs
    out_dirs = [Path(p) if os.path.isabs(p) else (args.src / p) for p in args.out]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)

    files = list_matching_files(args.src, args.recursive)
    if not files:
        print("No matching files found (looking for *Lo.npz or *Hi.npz).")
        return

    print(f"Found {len(files)} matching files.")

    if args.by_symbol:
        groups = group_by_symbol(files)
        symbols = list(groups.keys())
        print(f"Distributing {len(symbols)} symbols ({len(files)} files total).")

        if args.random:
            # Quota based on number of symbols
            sym_buckets = quota_split(symbols, args.pct)
        else:
            # Deterministic proportional slicing
            random.shuffle(symbols)
            total = len(symbols)
            cuts = [
                math.floor(total * args.pct[0] / 100.0),
                math.floor(total * (args.pct[0] + args.pct[1]) / 100.0),
                total
            ]
            sym_buckets = [
                symbols[:cuts[0]],
                symbols[cuts[0]:cuts[1]],
                symbols[cuts[1]:],
            ]

        bucket_lists = [
            [f for sym in sym_buckets[0] for f in groups[sym]],
            [f for sym in sym_buckets[1] for f in groups[sym]],
            [f for sym in sym_buckets[2] for f in groups[sym]],
        ]
    else:
        if args.random:
            bucket_lists = quota_split(files[:], args.pct)  # quota by count of files
        else:
            random.shuffle(files)
            total = len(files)
            cuts = [
                math.floor(total * args.pct[0] / 100.0),
                math.floor(total * (args.pct[0] + args.pct[1]) / 100.0),
                total
            ]
            bucket_lists = [
                files[:cuts[0]],
                files[cuts[0]:cuts[1]],
                files[cuts[1]:],
            ]

    print(f"Planned per bucket: {[len(b) for b in bucket_lists]}")
    move_batch(bucket_lists, out_dirs, commit=args.commit)

    if not args.commit:
        print("\nDry run only. Re-run with --commit to actually move files.")

if __name__ == "__main__":
    main()
