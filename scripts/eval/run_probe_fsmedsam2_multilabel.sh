#!/usr/bin/env bash
# Multi-label analog of the single-label FS_MedSAM2 confidence probe.
# Runs probe_fsmedsam2_confidence.py once per foreground class of any dataset,
# then concatenates into one combined CSV so pooled/per-class stats can be
# compared apples-to-apples against the multi-support method's scores.csv.
#
# No edits to third_party/ or probe_fsmedsam2_confidence.py -- pure driver.
#
# Usage (override via env vars):
#   DATA_DIR=data/datasets/AMOS/processed/T2 \
#   LABELS="LIVER:1 RK:2 LK:3 SPLEEN:4" \
#   OUT_DIR=results/fsmedsam2_probe/multilabel_amos \
#   bash scripts/eval/run_probe_fsmedsam2_multilabel.sh
set -euo pipefail

PYTHON=.venv/bin/python
DATA_DIR="${DATA_DIR:-data/datasets/MRI_muscle_2/processed/WATER}"
CKPT="${CKPT:-third_party/MedSAM2/checkpoints/sam2.1_hiera_tiny.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1_hiera_t512.yaml}"
DEVICE="${DEVICE:-cuda}"
OUT_DIR="${OUT_DIR:-results/fsmedsam2_probe/multilabel_mri_muscle_2}"
SEED="${SEED:-42}"
MAX_SCANS="${MAX_SCANS:-0}"

# space-separated name:label_val pairs; default = MRI_muscle_2's 6 classes
# (MRI_MUSCLE_2_LABEL_NAMES / RAW_TO_PROJECT_LABEL). Override via LABELS env var.
LABELS="${LABELS:-QF:1 HS:2 SA:3 GR:4 AD:5 GLUT:6}"

mkdir -p "$OUT_DIR"

for entry in $LABELS; do
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
for entry in $LABELS; do
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
