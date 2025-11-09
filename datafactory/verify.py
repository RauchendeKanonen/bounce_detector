#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
verify_npz.py

Check integrity of datafactory.npz files:
- prints start/end timestamps for every symbol
- detects missing quotes/trades
- verifies that all symbols share the same time window
- works with both single-symbol ("quotes"/"trades") and multi-symbol files
- works with sampled files ("sampled_X")
"""

import argparse
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

# ----------------------------------------------------------------------
def format_timestamp(ts: int) -> str:
    """Convert UTC epoch → US/Eastern human readable string"""
    if ts is None or ts <= 0:
        return "N/A"
    dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    return dt_et.strftime("%d.%m.%Y %H:%M:%S %Z")


def load_npz(path: str):
    """Same loader as datafactory – returns dicts keyed by symbol"""
    data = np.load(path, allow_pickle=True)
    meta_obj = data["meta"][0]
    meta = dict(meta_obj) if isinstance(meta_obj, dict) else meta_obj.item()

    # --- detect single-symbol mode ---
    if "quotes" in data and "trades" in data:
        sym = meta.get("symbol", "DEFAULT")
        return {sym: data["quotes"]}, {sym: data["trades"]}, meta

    # --- multi-symbol mode ---
    symbols = meta.get("symbols", [])
    quotes = {}
    trades = {}
    for sym in symbols:
        q_key = f"quotes_{sym}"
        t_key = f"trades_{sym}"
        if q_key not in data:
            raise KeyError(f"Missing {q_key}")
        if t_key not in data:
            raise KeyError(f"Missing {t_key}")
        quotes[sym] = data[q_key]
        trades[sym] = data[t_key]
    return quotes, trades, meta


def verify_raw_file(path: Path):
    print(f"\nVerifying RAW file: {path.name}")
    try:
        quotes_dict, trades_dict, meta = load_npz(str(path))
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return

    print(f"  Symbols: {', '.join(quotes_dict.keys())}")
    print(f"  Meta window: {meta.get('start_et')} → {meta.get('end_et')}")

    # collect per-symbol ranges
    ranges = []
    for sym in quotes_dict:
        q = quotes_dict[sym]
        t = trades_dict[sym]
        q_start = int(q["time"][0]) if q.size else None
        q_end   = int(q["time"][-1]) if q.size else None
        t_start = int(t["time"][0]) if t.size else None
        t_end   = int(t["time"][-1]) if t.size else None

        start = q_start if q_start is not None else t_start
        end   = q_end   if q_end   is not None else t_end

        ranges.append((sym, start, end, q.size, t.size))

    # print table
    print("  Symbol      | Quotes   | Trades   | First tick (ET)       | Last tick (ET)")
    print("  ------------+----------+----------+----------------------+----------------------")
    for sym, start, end, nq, nt in ranges:
        print(f"  {sym:11} | {nq:8} | {nt:8} | {format_timestamp(start):20} | {format_timestamp(end):20}")

    # check consistency
    starts = [r[1] for r in ranges if r[1]]
    ends   = [r[2] for r in ranges if r[2]]
    if starts and len(set(starts)) == 1 and ends and len(set(ends)) == 1:
        print("  All symbols perfectly aligned")
    else:
        print("  WARNING: symbols have different time ranges!")


def verify_sampled_file(path: Path):
    print(f"\nVerifying SAMPLED file: {path.name}")
    data = np.load(path, allow_pickle=True)

    sampled_keys = [k for k in data.files if k.startswith("sampled_")]
    if not sampled_keys:
        print("  No sampled_* arrays found")
        return

    print(f"  Found {len(sampled_keys)} sampled arrays")

    # extract symbol from key
    ranges = []
    for key in sampled_keys:
        sym = key.replace("sampled_", "")
        arr = data[key]
        t_left = arr["t_left"]
        t_right = arr["t_right"]
        ranges.append((sym, int(t_left[0]), int(t_right[-1]), arr.shape[0]))

    print("  Symbol      | Bins     | First bin (ET)        | Last bin (ET)")
    print("  ------------+----------+----------------------+----------------------")
    for sym, first, last, nbins in ranges:
        print(f"  {sym:11} | {nbins:8} | {format_timestamp(first):20} | {format_timestamp(last):20}")

    # check alignment
    firsts = [r[1] for r in ranges]
    lasts  = [r[2] for r in ranges]
    if len(set(firsts)) == 1 and len(set(lasts)) == 1:
        print("  All sampled arrays perfectly aligned")
    else:
        print("  WARNING: sampled arrays have different time grids!")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify datafactory.npz integrity")
    parser.add_argument("files", nargs="+", help="One or more .npz files (raw or sampled)")
    parser.add_argument("--sampled", action="store_true", help="Treat files as sampled (check sampled_* arrays)")
    args = parser.parse_args()

    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"File not found: {f}")
            continue

        if args.sampled:
            verify_sampled_file(path)
        else:
            # auto-detect: if file contains "sampled_" → sampled, else raw
            try:
                quick = np.load(path, allow_pickle=True)
                if any(k.startswith("sampled_") for k in quick.files):
                    verify_sampled_file(path)
                else:
                    verify_raw_file(path)
            except Exception as e:
                print(f"Cannot open {path.name}: {e}")
