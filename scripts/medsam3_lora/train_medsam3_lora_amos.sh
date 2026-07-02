#!/usr/bin/env bash
# Train MedSAM3's LoRA on AMOS22-MRI (T2 only, this project's data) --
# run scripts/medsam3_lora/prepare_amos_coco.sh first.
# Needs CUDA. Output: outputs/medsam3_lora_amos/best_model/lora_weights.pt
# (save_lora_only: true in configs/medsam3_lora_amos.yaml), pass that as
# --medsam3_weights to scripts/eval/eval_medsam3.py for eval on CHAOS.
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-0}"   # GPU id(s), e.g. DEVICE="0 1" for multi-GPU

echo "=== training MedSAM3 LoRA on AMOS T2 (configs/medsam3_lora_amos.yaml) ==="
$PYTHON third_party/MedSAM3/train_sam3_lora_native.py \
  --config configs/medsam3_lora_amos.yaml \
  --device $DEVICE
