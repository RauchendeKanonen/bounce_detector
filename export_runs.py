#!/usr/bin/env python3
import os
import json
import argparse
import csv
from pathlib import Path
from typing import Dict, Any

def collect_values_from_dir(d: Path) -> Dict[str, float]:
    """Read all *.json files in directory d (non-recursive) and collect top-level float/int values.
    If the same key appears in multiple files, the last one read wins."""
    values: Dict[str, float] = {}
    for entry in sorted(d.iterdir()):
        if entry.is_file() and entry.suffix.lower() == ".json":
            try:
                with entry.open("r", encoding="utf-8") as f:
                    data: Any = json.load(f)
                if isinstance(data, dict):
                    for k, v in data.items():
                        if isinstance(v, (int, float)):
                            values[k] = float(v)
                        else:
                            # try to coerce strings like "1.23"
                            if isinstance(v, str):
                                try:
                                    values[k] = float(v.strip())
                                except ValueError:
                                    pass  # ignore non-numeric
                # ignore non-dict JSON structures
            except Exception as e:
                print(f"Warning: couldn't read {entry}: {e}")
    return values

def main():
    ap = argparse.ArgumentParser(description="Collect float values from JSON files in immediate subdirectories into a CSV.")
    ap.add_argument("base", nargs="?", default=".", help="Base directory (default: current directory)")
    ap.add_argument("-o", "--output", default="summary.csv", help="Output CSV path (default: summary.csv)")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.is_dir():
        raise SystemExit(f"Base path is not a directory: {base}")

    # Find immediate subdirectories
    subdirs = [p for p in sorted(base.iterdir()) if p.is_dir()]
    if not subdirs:
        print("No subdirectories found. Nothing to do.")
        return

    # Collect per-dir values and all keys
    per_dir: Dict[str, Dict[str, float]] = {}
    all_keys = set()
    for d in subdirs:
        vals = collect_values_from_dir(d)
        if vals:
            per_dir[d.name] = vals
            all_keys.update(vals.keys())

    if not all_keys:
        print("No numeric values found in JSON files.")
        return

    # Stable, readable column order: sorted keys
    headers = list(all_keys)
    headers.sort(key=lambda x: (x.endswith("score"), x))
    headers = ["directory"] + headers

    # Write CSV
    out_path = Path(args.output)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for dname in sorted(per_dir.keys()):
            row = [dname]
            vals = per_dir[dname]
            for k in headers[1:]:
                row.append(vals.get(k, ""))  # empty if missing
            writer.writerow(row)

    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
