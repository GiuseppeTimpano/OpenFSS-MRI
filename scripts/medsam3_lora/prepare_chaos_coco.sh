#!/usr/bin/env bash
# CHAOS T1+T2 NIfTI -> COCO (scripts/medsam3_lora/build_coco_dataset.py), for
# third_party/MedSAM3/train_sam3_lora_native.py. Patient-level train/valid
# split (get_fold_ids), fold 0 of 5 held out as valid (~20%).
set -euo pipefail

PYTHON=.venv/bin/python

for seq in T1 T2; do
  echo "=== CHAOS $seq -> COCO ==="
  PYTHONPATH=. $PYTHON scripts/medsam3_lora/build_coco_dataset.py \
    --data_dir "data/datasets/CHAOS/processed/$seq" \
    --out_dir  "data/datasets/CHAOS_coco/$seq" \
    --labels 1 2 3 4 \
    --label_names BG LIVER RK LK SPLEEN \
    --fold 0 --n_folds 5
done

echo "Done. Next: scripts/medsam3_lora/train_medsam3_lora_chaos.sh"
