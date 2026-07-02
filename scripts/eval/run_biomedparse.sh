#!/usr/bin/env bash
# Full BiomedParse eval: CirrMRI (LIVER) + CHAOS (LIVER, RK, LK, SPLEEN), T1+T2.
# No --limit -> full dataset. Adjust DEVICE/SAVE_DIR as needed.
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results/biomedparse}"
SAVE_TOPK="${SAVE_TOPK:-1}"   # per class: N best + N worst nii.gz saved; 0 = CSV only

run() {
  local dataset_dir=$1 seq=$2 out=$3
  shift 3
  local out_dir="$SAVE_DIR/$out"
  mkdir -p "$out_dir"
  echo "=== BiomedParse: $dataset_dir ($seq) labels: $* ==="
  PYTHONPATH=. $PYTHON scripts/eval/eval_biomedparse.py \
    --target_data_dir "$dataset_dir" \
    --test_label "$@" \
    --device "$DEVICE" \
    --save_dir "$out_dir" \
    --save_topk "$SAVE_TOPK" \
    2>&1 | tee "$out_dir/run.log"
}

# CirrMRI: liver only
run data/datasets/CIRRMR/processed/T1 T1 cirrmri_t1 1
run data/datasets/CIRRMR/processed/T2 T2 cirrmri_t2 1

# CHAOS: liver + kidneys + spleen
run data/datasets/CHAOS/processed/T1 T1 chaos_t1 1 2 3 4
run data/datasets/CHAOS/processed/T2 T2 chaos_t2 1 2 3 4

echo "Done. Logs in $SAVE_DIR/"
