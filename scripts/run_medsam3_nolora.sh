#!/usr/bin/env bash
# Plain pretrained SAM3 baseline (--no_lora): NO MedSAM3-v1 LoRA weights applied.
# Same datasets/labels as run_medsam3.sh -- diagnostic to tell apart
# "grounding head can't localize small organs" (architectural, would show up
# here too) from "LoRA weights specifically hurt RK/LK/SPLEEN"
# (training-data/overfit, would NOT show up here) -- see models/medsam3_adapter.py
# build_medsam3_lora_model() docstring.
# No --limit -> full dataset. Adjust DEVICE/SAVE_DIR as needed.
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-results/medsam3_nolora}"
SAVE_TOPK="${SAVE_TOPK:-1}"   # per class: N best + N worst nii.gz saved; 0 = CSV only

run() {
  local dataset_dir=$1 seq=$2 out=$3
  shift 3
  local out_dir="$SAVE_DIR/$out"
  mkdir -p "$out_dir"
  echo "=== MedSAM3 (no LoRA, plain SAM3): $dataset_dir ($seq) labels: $* ==="
  PYTHONPATH=. $PYTHON eval_medsam3.py \
    --target_data_dir "$dataset_dir" \
    --test_label "$@" \
    --device "$DEVICE" \
    --save_dir "$out_dir" \
    --save_topk "$SAVE_TOPK" \
    --no_lora \
    2>&1 | tee "$out_dir/run.log"
}

# CirrMRI: liver only
run data/datasets/CIRRMR/processed/T1 T1 cirrmri_t1 1
run data/datasets/CIRRMR/processed/T2 T2 cirrmri_t2 1

# CHAOS: liver + kidneys + spleen
run data/datasets/CHAOS/processed/T1 T1 chaos_t1 1 2 3 4
run data/datasets/CHAOS/processed/T2 T2 chaos_t2 1 2 3 4

echo "Done. Logs in $SAVE_DIR/"
