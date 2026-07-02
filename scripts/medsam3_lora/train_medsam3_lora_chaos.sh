#!/usr/bin/env bash
# Retrain MedSAM3-v1's LoRA on CHAOS (this project's data, not the paper's
# proprietary set) -- run scripts/medsam3_lora/prepare_chaos_coco.sh first.
# Needs CUDA. Output: outputs/medsam3_lora_chaos_{t1,t2}/best_model/lora_weights.pt
# (save_lora_only: true in configs/medsam3_lora_chaos_*.yaml), pass that as
# --medsam3_weights to scripts/eval/eval_medsam3.py for zero-shot eval on e.g. AMOS.
set -euo pipefail

PYTHON=.venv/bin/python
DEVICE="${DEVICE:-0}"   # GPU id(s), e.g. DEVICE="0 1" for multi-GPU
SEQ="${1:-T1}"          # T1 or T2

case "$SEQ" in
  T1) CONFIG=configs/medsam3_lora_chaos_t1.yaml ;;
  T2) CONFIG=configs/medsam3_lora_chaos_t2.yaml ;;
  *) echo "usage: $0 [T1|T2]"; exit 1 ;;
esac

echo "=== training MedSAM3 LoRA on CHAOS $SEQ ($CONFIG) ==="
$PYTHON third_party/MedSAM3/train_sam3_lora_native.py \
  --config "$CONFIG" \
  --device $DEVICE
