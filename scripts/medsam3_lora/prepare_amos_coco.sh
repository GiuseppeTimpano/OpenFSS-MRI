#!/usr/bin/env bash
# AMOS22-MRI T2 (already remapped to CHAOS label convention, see
# data/datasets/amos.py:AMOS_TO_CHAOS) NIfTI -> COCO
# (scripts/medsam3_lora/build_coco_dataset.py), for
# third_party/MedSAM3/train_sam3_lora_native.py. Patient-level train/valid
# split (get_fold_ids), fold 0 of 5 held out as valid (~20%).
set -euo pipefail

PYTHON=.venv/bin/python

echo "=== AMOS T2 -> COCO ==="
PYTHONPATH=. $PYTHON scripts/medsam3_lora/build_coco_dataset.py \
  --data_dir "data/datasets/AMOS/processed/T2" \
  --out_dir  "data/datasets/AMOS_coco/T2" \
  --labels 1 2 3 4 \
  --label_names BG LIVER RK LK SPLEEN \
  --fold 0 --n_folds 5

echo "Done. Next: scripts/medsam3_lora/train_medsam3_lora_amos.sh"
