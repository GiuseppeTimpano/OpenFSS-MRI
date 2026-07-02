#!/usr/bin/env bash
# End-to-end: AMOS22-MRI T2 -> COCO, then MedSAM3 LoRA training on it.
# Skips COCO conversion if data/datasets/AMOS_coco/T2 already exists.
# Needs CUDA for training. Output: outputs/medsam3_lora_amos/best_model/lora_weights.pt
set -euo pipefail

DEVICE="${DEVICE:-0}"   # GPU id(s), e.g. DEVICE="0 1" for multi-GPU
COCO_DIR="data/datasets/AMOS_coco/T2"

if [ -d "$COCO_DIR" ]; then
  echo "=== $COCO_DIR already exists, skipping COCO conversion ==="
else
  bash scripts/medsam3_lora/prepare_amos_coco.sh
fi

DEVICE="$DEVICE" bash scripts/medsam3_lora/train_medsam3_lora_amos.sh
