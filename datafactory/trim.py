#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
trim_npz.py

Trim a datafactory.npz file to the *smallest overlapping time window*
across all symbols (raw or sampled).

Usage:
    python trim_npz.py input.npz output.npz
    python trim_npz.py input.npz --dry-run
"""

import argparse
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

# ----------------------------------------------------------------------
DE_FMT_FULL = "%d.%m.%Y %H:%M:%S"

def format_et(ts: int) -> str:
    if ts is None:
        return "N/A"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone(ZoneInfo("America/New_York"))
    return dt.strftime(DE_FMT_FULL)

# ----------------------------------------------------------------------
def load_raw(path: str):
    data = np.load(path, allow_pickle=True)
    meta = dict(data["meta"][0])

    # Single-symbol mode?
    if "quotes" in data and "trades" in data:
        sym = meta.get("symbol", "DEFAULT")
        return {sym: data["quotes"]}, {sym: data["trades"]}, meta

    # Multi-symbol
    symbols = meta.get("symbols", [])
    quotes = {s: data[f"quotes_{s}"] for s in symbols}
    trades = {s: data[f"trades_{s}"] for s in symbols}
    return quotes, trades, meta

def load_sampled(path: str):
    data = np.load(path, allow_pickle=True)
    meta = dict(data["meta"][0])
    sampled = {}
    for k in data.files:
        if k.startswith("sampled_"):
            sym = k[len("sampled_"):]
            sampled[sym] = data[k]
    return sampled, meta

# ----------------------------------------------------------------------
def find_common_window_raw(quotes_dict, trades_dict):
    starts = []
    ends = []

    for sym in quotes_dict:
        q = quotes_dict[sym]
        t = trades_dict[sym]
        if q.size:
            starts.append(int(q["time"][0]))
            ends.append(int(q["time"][-1]))
        if t.size:
            starts.append(int(t["time"][0]))
            ends.append(int(t["time"][-1]))

    if not starts or not ends:
        raise ValueError("No data found in any symbol")

    common_start = max(starts)
    common_end = min(ends)
    if common_start >= common_end:
        raise ValueError("No overlapping time window")

    return common_start, common_end

def find_common_window_sampled(sampled_dict):
    starts = [int(arr["t_left"][0]) for arr in sampled_dict.values()]
    ends   = [int(arr["t_right"][-1]) for arr in sampled_dict.values()]

    common_start = max(starts)
    common_end = min(ends)
    if common_start >= common_end:
        raise ValueError("No overlapping time window")

    return common_start, common_end

# ----------------------------------------------------------------------
def trim_raw(quotes_dict, trades_dict, start, end):
    trimmed_q = {}
    trimmed_t = {}

    for sym in quotes_dict:
        q = quotes_dict[sym]
        t = trades_dict[sym]

        # Trim quotes
        if q.size:
            mask = (q["time"] >= start) & (q["time"] <= end)
            trimmed_q[sym] = q[mask]
        else:
            trimmed_q[sym] = q.copy()

        # Trim trades
        if t.size:
            mask = (t["time"] >= start) & (t["time"] <= end)
            trimmed_t[sym] = t[mask]
        else:
            trimmed_t[sym] = t.copy()

    return trimmed_q, trimmed_t

def trim_sampled(sampled_dict, start, end):
    trimmed = {}
    for sym, arr in sampled_dict.items():
        # Find bin indices
        left_in = np.searchsorted(arr["t_left"], start, side="left")
        right_ex = np.searchsorted(arr["t_right"], end, side="right")

        trimmed_arr = arr[left_in:right_ex]
        if trimmed_arr.size == 0:
            raise ValueError(f"Symbol {sym} has no bins in common window")
        trimmed[sym] = trimmed_arr
    return trimmed

# ----------------------------------------------------------------------
def save_raw(path: str, quotes, trades, meta, start, end):
    save_args = {}
    symbols = list(quotes.keys())

    if len(symbols) == 1:
        sym = symbols[0]
        save_args["quotes"] = quotes[sym]
        save_args["trades"] = trades[sym]
        meta["symbol"] = sym
        if "symbols" in meta:
            del meta["symbols"]
    else:
        for sym in symbols:
            save_args[f"quotes_{sym}"] = quotes[sym]
            save_args[f"trades_{sym}"] = trades[sym]
        meta["symbols"] = symbols

    # Update meta
    meta["start_et"] = format_et(start)
    meta["end_et"] = format_et(end)
    meta["start_epoch"] = start
    meta["end_epoch"] = end

    save_args["meta"] = np.array([meta], dtype=object)
    np.savez_compressed(path, **save_args)

def save_sampled(path: str, sampled, meta, start, end):
    save_args = {}
    for sym, arr in sampled.items():
        save_args[f"sampled_{sym}"] = arr
        save_args[f"info_{sym}"] = np.array([{
            "dt_sec": meta.get("dt_sec", 0),
            "n_bins": arr.shape[0],
            "t0": int(arr["t_left"][0]),
            "t1": int(arr["t_right"][-1])
        }], dtype=object)

    meta["start_et"] = format_et(start)
    meta["end_et"] = format_et(end)
    save_args["meta"] = np.array([meta], dtype=object)
    np.savez_compressed(path, **save_args)

# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Trim NPZ to smallest common time window")
    parser.add_argument("input", help="Input .npz file")
    parser.add_argument("output", nargs="?", help="Output .npz file (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Show trim plan, don't write")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"{in_path} not found")

    # Auto-detect type
    with np.load(in_path, allow_pickle=True) as f:
        is_sampled = any(k.startswith("sampled_") for k in f.keys())

    print(f"Loading {'SAMPLED' if is_sampled else 'RAW'} file: {in_path.name}")

    if is_sampled:
        sampled_dict, meta = load_sampled(str(in_path))
        print(f"  Found {len(sampled_dict)} sampled symbols: {', '.join(sampled_dict.keys())}")
        start, end = find_common_window_sampled(sampled_dict)
    else:
        quotes_dict, trades_dict, meta = load_raw(str(in_path))
        symbols = list(quotes_dict.keys())
        print(f"  Found {len(symbols)} symbols: {', '.join(symbols)}")
        start, end = find_common_window_raw(quotes_dict, trades_dict)

    print(f"\nCommon window:")
    print(f"  Start: {format_et(start)}")
    print(f"  End:   {format_et(end)}")
    print(f"  Duration: {(end - start) / 3600:.2f} hours")

    if args.dry_run:
        print("\n--dry-run: No file written.")
        return

    out_path = Path(args.output or in_path.stem + "_trimmed.npz")

    if is_sampled:
        trimmed = trim_sampled(sampled_dict, start, end)
        save_sampled(str(out_path), trimmed, meta, start, end)
    else:
        trimmed_q, trimmed_t = trim_raw(quotes_dict, trades_dict, start, end)
        save_raw(str(out_path), trimmed_q, trimmed_t, meta, start, end)

    print(f"\nTrimmed file saved: {out_path}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
