#!/usr/bin/env python3
"""
Event-based labeling (simple zigzag):

For each NPZ episode file with a structured array `sampled`,
1) detect swing LOWs and HIGHs using a percent zigzag threshold,
2) for every LOW -> next HIGH, compute gain,
3) label the LOW as bounce (1) if gain >= U, else 0, all other rows = -1.

Outputs a new `<input>_labeled_events.npz` per input with:
  - sampled            (unchanged)
  - info, meta         (passthrough)
  - labels             (int8, length N; -1 elsewhere, 0/1 at LOW indices that have a following HIGH)
  - valid_idx          (int64 indices where labels are 0 or 1; i.e., LOWs with a following HIGH)
  - events             (structured array per low→high pair: indices, times, prices, gain)
  - params             (dict of the parameters used)

No intra-file splitting is performed (episode=whole file). You can choose
which files are train/val/test outside this step (episode-level split).

Usage:
  python label_events_zigzag.py \
      --in ep1.npz ep2.npz \
      --pct 0.005 --gain 0.003 --min_sep 3

Parameters:
  --pct      : zigzag reversal threshold as fraction (0.005 = 0.5%)
  --gain     : required low→next-high gain to call it a bounce (0.003 = 0.3%)
  --min_sep  : minimum bin separation between successive extrema (debounce)

Notes:
- Works purely on `mid` (from sampled). It ignores rows where mid/spread/bid/ask are invalid.
- Does not average anything; uses snapshot values.
- Tail is left open (no forced last extremum at the end to avoid look-ahead).
"""

from __future__ import annotations
import argparse
import os
import numpy as np
from typing import List, Tuple, Dict

# ------------------- utilities -------------------

def load_npz(path: str):
    npz = np.load(path, allow_pickle=True)
    sampled = npz["sampled"]
    info = npz["info"].item() if npz["info"].dtype == object else npz["info"]
    meta = npz["meta"].item() if npz["meta"].dtype == object else npz["meta"]
    return sampled, info, meta


def detect_missing(sampled: np.ndarray) -> np.ndarray:
    """Row is invalid if mid/spread/bid/ask is NaN or non-positive / spread<=0."""
    mid = sampled["mid"]
    spread = sampled["spread"]
    bid = sampled["bid"]
    ask = sampled["ask"]
    miss = (
        np.isnan(mid) | np.isnan(spread) | np.isnan(bid) | np.isnan(ask)
        | (spread <= 0.0) | (bid <= 0.0) | (ask <= 0.0)
    )
    return miss


def compress_same_kind_extrema(seq: List[Tuple[int, str]], mid: np.ndarray) -> List[Tuple[int, str]]:
    """
    Merge consecutive extrema of the same kind:
      - for LOW runs, keep the lowest price,
      - for HIGH runs, keep the highest price.
    Ensures the output alternates low/high/low/high...
    """
    if not seq:
        return []
    out: List[Tuple[int, str]] = []
    i, n = 0, len(seq)
    while i < n:
        kind = seq[i][1]
        best_idx = seq[i][0]
        best_val = mid[best_idx]
        j = i + 1
        while j < n and seq[j][1] == kind:
            idx = seq[j][0]
            val = mid[idx]
            if kind == "low":
                if val < best_val:
                    best_idx, best_val = idx, val
            else:  # "high"
                if val > best_val:
                    best_idx, best_val = idx, val
            j += 1
        out.append((best_idx, kind))
        i = j
    return out


def pair_low_to_next_high_robust(seq: List[Tuple[int, str]], mid: np.ndarray) -> List[Tuple[int, int]]:
    """
    Walk the (possibly noisy) extrema sequence and create one-to-one LOW→HIGH pairs.
    - If multiple LOWs happen before any HIGH, keep the *deeper* LOW (lower price).
    - Each HIGH is consumed once.
    """
    pairs: List[Tuple[int, int]] = []
    curr_low: int | None = None
    for idx, kind in seq:
        if kind == "low":
            if curr_low is None or mid[idx] < mid[curr_low]:
                curr_low = idx        # keep deeper low
        else:  # kind == "high"
            if curr_low is not None and idx > curr_low:
                pairs.append((curr_low, idx))
                curr_low = None       # consume low; high used once
    return pairs


# ------------------- zigzag detector -------------------

def zigzag_extrema_sequence(mid: np.ndarray, pct: float, min_sep: int) -> List[Tuple[int, str]]:
    """
    Return a sequence of extrema as a list of (index, kind) where kind ∈ {"low","high"}.
    This implementation enforces alternation by design and a minimum separation.

    pct     : fractional reversal threshold (e.g., 0.005 = 0.5%).
    min_sep : minimum number of bins between successive extrema (debounce).
    """
    n = len(mid)
    if n == 0:
        return []

    seq: List[Tuple[int, str]] = []
    last_ext_i = 0
    last_ext_p = mid[0]
    direction = 0  # 0=unknown, +1=up leg (tracking highs), -1=down leg (tracking lows)
    last_mark_i = -min_sep - 1  # so the first can pass min_sep check

    for i in range(1, n):
        p = mid[i]
        if not (p > 0) or np.isnan(p):
            continue
        change = (p - last_ext_p) / last_ext_p if last_ext_p > 0 else 0.0

        if direction >= 0:  # up/unknown: track highs, look for down reversal
            if p > last_ext_p:
                last_ext_p = p
                last_ext_i = i
                direction = +1
            elif direction != 0 and change <= -pct:
                # down reversal confirmed -> previous was HIGH
                if i - last_mark_i >= min_sep:
                    seq.append((last_ext_i, "high"))
                    last_mark_i = last_ext_i
                last_ext_p = p
                last_ext_i = i
                direction = -1

        if direction <= 0:  # down/unknown: track lows, look for up reversal
            if p < last_ext_p:
                last_ext_p = p
                last_ext_i = i
                direction = -1
            elif direction != 0 and change >= pct:
                # up reversal confirmed -> previous was LOW
                if i - last_mark_i >= min_sep:
                    seq.append((last_ext_i, "low"))
                    last_mark_i = last_ext_i
                last_ext_p = p
                last_ext_i = i
                direction = +1

    # Do NOT close the last leg with a terminal extremum to avoid look-ahead bias on tail.
    # Ensure alternation: if first two have same kind (rare), drop the first.
    if len(seq) >= 2 and seq[0][1] == seq[1][1]:
        seq = seq[1:]
    return seq


# ------------------- labeling -------------------

def label_bounces(mid: np.ndarray, pairs: List[Tuple[int, int]], U: float,
                  Type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Point labels at LOW indices only (0/1). No spans.
    Returns: labels (int8, len N), valid_idx (LOW indices used), gains (float array aligned to valid_idx).
    :param Type:
    """
    N = len(mid)
    labels = np.full(N, 0, dtype=np.int8)
    valid_idx: List[int] = []
    gains: List[float] = []

    for li, hi in pairs:
        if not (0 <= li < N and 0 <= hi < N and hi > li):
            continue
        m0, mh = mid[li], mid[hi]
        if not (m0 > 0 and mh > 0) or np.isnan(m0) or np.isnan(mh):
            continue
        gain = (mh - m0) / m0
        if Type == "Lo":
            labels[li] = 1 if gain >= U else 0   # <- point label at LOW only
            valid_idx.append(li)
        if Type == "Hi":
            labels[hi] = 1 if gain >= U else 0  # <- point label at LOW only
            valid_idx.append(hi)


        gains.append(gain)

    return labels, np.array(valid_idx, np.int64), np.array(gains, np.float64)


def make_events_array(sampled: np.ndarray,
                      pairs: List[Tuple[int, int]],
                      gains: np.ndarray) -> np.ndarray:
    """Build a structured array of events with indices, times, prices, and gain."""
    t_left = sampled["t_left"]
    mid = sampled["mid"]
    dt = np.dtype([
        ("low_idx", "<i8"), ("high_idx", "<i8"),
        ("t_low", "<i8"), ("t_high", "<i8"),
        ("m_low", "<f8"), ("m_high", "<f8"),
        ("gain", "<f8")
    ])
    out = np.zeros(len(gains), dtype=dt)
    for k, ((li, hi), g) in enumerate(zip(pairs, gains)):
        out[k]["low_idx"] = int(li)
        out[k]["high_idx"] = int(hi)
        out[k]["t_low"] = int(t_left[li])
        out[k]["t_high"] = int(t_left[hi])
        out[k]["m_low"] = float(mid[li])
        out[k]["m_high"] = float(mid[hi])
        out[k]["gain"] = float(g)
    return out


# ------------------- main -------------------

def process_file(in_path: str, pct: float, gain_thr: float, min_sep: int, Type: str) -> str:
    sampled, info, meta = load_npz(in_path)
    mid = sampled["mid"].astype(np.float64)
    missing = detect_missing(sampled)

    # Build a working mid series for the zigzag by forward/back filling invalids
    mid_work = mid.copy()
    invalid = missing | np.isnan(mid_work) | ~(mid_work > 0)
    if invalid.any():
        # forward fill
        for i in range(1, len(mid_work)):
            if invalid[i] and not invalid[i - 1]:
                mid_work[i] = mid_work[i - 1]
        # back fill head if needed
        for i in range(len(mid_work) - 2, -1, -1):
            if invalid[i] and not invalid[i + 1]:
                mid_work[i] = mid_work[i + 1]

    # Detect extrema and pair LOW→next HIGH robustly
    seq = zigzag_extrema_sequence(mid_work, pct=pct, min_sep=min_sep)
    seq = compress_same_kind_extrema(seq, mid)
    pairs = pair_low_to_next_high_robust(seq, mid)

    # Point labels at LOWs
    labels, valid_idx, gains = label_bounces(mid, pairs, U=gain_thr, Type=Type)

    # Remove any labels/pairs that land on missing rows (safety)
    if len(valid_idx):
        keep_mask = ~missing[valid_idx]
        valid_idx = valid_idx[keep_mask]
        gains = gains[keep_mask]
        # Keep only pairs whose LOW is kept
        low_set = set(valid_idx.tolist())
        pairs = [p for p in pairs if p[0] in low_set]
        # Clear labels at removed lows
        to_clear = [li for (li, _hi) in pairs if li not in low_set]
        if to_clear:
            labels[np.array(to_clear, dtype=np.int64)] = -1

    # Build events
    events = make_events_array(sampled, pairs, gains)

    params: Dict = dict(
        method="zigzag_low_to_next_high",
        pct=pct,
        gain=gain_thr,
        min_sep=min_sep,
        notes="Labels at LOW indices only. 1 if next HIGH gain >= gain, else 0. Others -1. or revers according to Type"
    )

    base, _ = os.path.splitext(in_path)
    out_path = base + "_labeled_events_" + Type + ".npz"

    if len(pairs) < 4:
        print("skipping save due to no or too little pairs")
        return ""
    np.savez_compressed(
        out_path,
        sampled=sampled,
        info=info,
        meta=meta,
        labels=labels,
        valid_idx=valid_idx,
        events=events,
        params=params,
    )

    # Brief console summary
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    inv = int((labels == -1).sum())
    print(f"[OK] {in_path} → {out_path} | labels: +1={pos}, 0={neg}, -1={inv} | valid lows={len(valid_idx)} | pairs={len(pairs)}")



    return out_path


def main():
    ap = argparse.ArgumentParser(description="Event-based labeling via simple zigzag (low→next-high).")
    ap.add_argument("--in", dest="inputs", nargs="+", required=True, help="One or more NPZ episode files.")
    ap.add_argument("--pct", type=float, default=0.005, help="Zigzag threshold fraction (e.g., 0.005 = 0.5%).")
    ap.add_argument("--gain", type=float, default=0.003, help="Gain threshold for low→next-high (e.g., 0.003 = 0.3%).")
    ap.add_argument("--min_sep", type=int, default=3, help="Minimum bin separation between consecutive extrema.")
    ap.add_argument("--Type", type=str, default="Lo", help="Lo -> labels for Lows Hi -> labels for Highs")

    args = ap.parse_args()

    for path in args.inputs:
        process_file(path, pct=args.pct, gain_thr=args.gain, min_sep=args.min_sep, Type=args.Type)


if __name__ == "__main__":
    main()
