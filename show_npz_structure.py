import numpy as np
import sys

def show_npz_structure(filename):
    try:
        with np.load(filename, allow_pickle=True) as data:
            print(f"Contents of '{filename}':")
            for key in data.files:
                arr = data[key]
                print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    except Exception as e:
        print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_npz_structure.py <file.npz>")
    else:
        show_npz_structure(sys.argv[1])
