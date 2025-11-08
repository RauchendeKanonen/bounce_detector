#!/bin/bash
# batch_sample.sh
# Converts all *_rth.npz raw tick files into sampled data using ib_ticks_pipeline_et.py

# Adjustable parameters
DT_SEC=1              # sampling interval in seconds
SCRIPT="datafactory.py"

# --- main ---
for infile in data/*_rth.npz; do
  # Skip if no files match
  [ -e "$infile" ] || { echo "No raw .npz files found."; exit 0; }

  # Build output filename
  base="${infile%_rth.npz}"
  outfile="${base}_sampled_${DT_SEC}s.npz"

  echo "Processing: $infile  →  $outfile"
  python "$SCRIPT" sample --infile "$infile" --dt-sec "$DT_SEC" --outfile "$outfile"

  if [ $? -eq 0 ]; then
    echo "✔ Done: $outfile"
  else
    echo "✖ Error processing $infile"
  fi
done

echo "All done."
