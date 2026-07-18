#!/usr/bin/env bash
# Multi-label analog of the single-label FS_MedSAM2 confidence probe.
# Runs probe_fsmedsam2_confidence.py once per muscle class (same 6 classes
# used by the multi-support eval: results/muscle_mri_2/mcvis*/scores.csv),
# then concatenates into one combined CSV so pooled/per-class stats can be
# compared apples-to-apples against the multi-support method.
#
# No edits to third_party/ or probe_fsmedsam2_confidence.py -- pure driver.
set -euo pipefail

PYTHON=.venv/bin/python
DATA_DIR="${DATA_DIR:-data/datasets/MRI_muscle_2/processed/WATER}"
CKPT="${CKPT:-third_party/MedSAM2/checkpoints/MedSAM2_latest.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1_hiera_t512.yaml}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/fsmedsam2_probe/multilabel_mri_muscle_2}"
SEED="${SEED:-42}"
MAX_SCANS="${MAX_SCANS:-0}"

mkdir -p "$OUT_DIR"

# name:label_val, matching MRI_MUSCLE_2_LABEL_NAMES / RAW_TO_PROJECT_LABEL
declare -a LABELS=(QF:1 HS:2 SA:3 GR:4 AD:5 GLUT:6)

for entry in "${LABELS[@]}"; do
  name="${entry%%:*}"
  val="${entry##*:}"
  echo "=== probing $name (label_val=$val) ==="
  PYTHONPATH=. $PYTHON -m scripts.eval.probe_fsmedsam2_confidence \
    --data_dir "$DATA_DIR" \
    --label_val "$val" \
    --label_name "$name" \
    --medsam2_ckpt "$CKPT" \
    --sam2_cfg "$SAM2_CFG" \
    --device "$DEVICE" \
    --out_csv "$OUT_DIR/${name}.csv" \
    --seed "$SEED" \
    --max_scans "$MAX_SCANS"
done

# merge into one combined CSV (header once)
combined="$OUT_DIR/all_labels.csv"
first=1
: > "$combined"
for entry in "${LABELS[@]}"; do
  name="${entry%%:*}"
  f="$OUT_DIR/${name}.csv"
  if [ "$first" -eq 1 ]; then
    cat "$f" > "$combined"
    first=0
  else
    tail -n +2 "$f" >> "$combined"
  fi
done

echo "Done. Per-label CSVs + combined in $OUT_DIR/"
