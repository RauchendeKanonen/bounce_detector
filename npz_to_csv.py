#!/usr/bin/env python3
# npz_to_csv.py
"""
Convert an NPZ file (from ib_ticks_pipeline) to a CSV.
Usage:
    python npz_to_csv.py input_file.npz [output_file.csv]
If output_file is not given, it will be derived automatically.
"""

import sys
import os
import numpy as np
import pandas as pd

def npz_to_csv(infile: str, outfile: str = None):
    if not os.path.exists(infile):
        print(f"Error: File not found: {infile}")
        sys.exit(1)

    # Derive output filename
    if outfile is None:
        outfile = os.path.splitext(infile)[0] + ".csv"

    # Load NPZ contents
    data = np.load(infile, allow_pickle=True)
    keys = list(data.keys())
    print(f"Loaded {infile}")
    print(f"Contains: {keys}")

    # Pick the main data array
    if "sampled" in keys:
        arr = data["sampled"]
    elif "quotes" in keys:
        arr = data["quotes"]
    elif "trades" in keys:
        arr = data["trades"]
    else:
        print("No recognizable data array found (expected sampled, quotes, or trades).")
        sys.exit(1)

    # Convert structured array to DataFrame
    df = pd.DataFrame(arr)

    # Save to CSV
    df.to_csv(outfile, index=False)
    print(f"Saved to {outfile} ({len(df)} rows)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python npz_to_csv.py input_file.npz [output_file.csv]")
        sys.exit(1)

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else None
    npz_to_csv(infile, outfile)
