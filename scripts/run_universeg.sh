#!/usr/bin/env bash
# Full UniverSeg eval (in-context, deployable): CirrMRI (LIVER), T1+T2.
# CHAOS excluded: UniverSeg was trained on CHAOS, so it's not a held-out set for this model.
# No --limit -> full dataset (support scan comes from the fold's own test split).
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-cuda}"       # cuda | cpu | mps
SUPP_IDX="${SUPP_IDX:-0}"      # index into the fold's test-id list for the support scan
N_PART="${N_PART:-3}"          # number of support FG slices (same default as test.py)
SAVE_DIR="${SAVE_DIR:-results/universeg}"
SAVE_TOPK="${SAVE_TOPK:-1}"    # per class: N best + N worst nii.gz saved; 0 = CSV only

run() {
  local dataset_dir=$1 seq=$2 out=$3
  shift 3
  local out_dir="$SAVE_DIR/$out"
  mkdir -p "$out_dir"
  echo "=== UniverSeg: $dataset_dir ($seq) labels: $* ==="
  PYTHONPATH=. $PYTHON eval_universeg.py \
    --target_data_dir "$dataset_dir" \
    --test_label "$@" \
    --supp_idx "$SUPP_IDX" \
    --n_part "$N_PART" \
    --device "$DEVICE" \
    --save_dir "$out_dir" \
    --save_topk "$SAVE_TOPK" \
    2>&1 | tee "$out_dir/run.log"
}

# CirrMRI: liver only
run data/datasets/CIRRMR/processed/T1 T1 cirrmri_t1 1
run data/datasets/CIRRMR/processed/T2 T2 cirrmri_t2 1

echo "Done. Logs in $SAVE_DIR/"
