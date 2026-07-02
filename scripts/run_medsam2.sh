#!/usr/bin/env bash
# Full MedSAM2 eval (oracle box): CirrMRI (LIVER), T1+T2.
# CHAOS excluded: MedSAM2 was trained on CHAOS, so it's not a held-out set for this model.
# No --limit -> full dataset. Override CKPT to point at sam2.1_hiera_tiny.pt for the
# SAM2-vanilla control run (see HANDOFF.md "SAM2 vanilla control").
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-cuda}"       # cuda | cpu | mps
CKPT="${CKPT:-third_party/MedSAM2/checkpoints/MedSAM2_latest.pt}"
SAM2_CFG="${SAM2_CFG:-configs/sam2.1_hiera_t512.yaml}"
PROMPT_MODE="${PROMPT_MODE:-perslice}"   # perslice (oracle upper bound) | key
SAVE_DIR="${SAVE_DIR:-results/medsam2}"
SAVE_TOPK="${SAVE_TOPK:-1}"    # per class: N best + N worst nii.gz saved; 0 = CSV only

run() {
  local dataset_dir=$1 seq=$2 out=$3
  shift 3
  local out_dir="$SAVE_DIR/$out"
  mkdir -p "$out_dir"
  echo "=== MedSAM2: $dataset_dir ($seq) labels: $* ==="
  PYTHONPATH=. $PYTHON eval_medsam2.py \
    --medsam2_ckpt "$CKPT" \
    --sam2_cfg "$SAM2_CFG" \
    --target_data_dir "$dataset_dir" \
    --test_label "$@" \
    --prompt_mode "$PROMPT_MODE" \
    --device "$DEVICE" \
    --save_dir "$out_dir" \
    --save_topk "$SAVE_TOPK" \
    2>&1 | tee "$out_dir/run.log"
}

# CirrMRI: liver only
run data/datasets/CIRRMR/processed/T1 T1 cirrmri_t1 1
run data/datasets/CIRRMR/processed/T2 T2 cirrmri_t2 1

echo "Done. Logs in $SAVE_DIR/"
