#!/usr/bin/env bash
# A/B self-refine su MedSAM2 support_bbox — MRI_muscle, label R_HS (6).
# SERVER Linux (CUDA). Confronta refine_iters=0 (baseline) vs =2 a parita' di seed/pairing.
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI
source .venv/bin/activate

export PYTHONPATH=.

CKPT=third_party/MedSAM2/checkpoints/MedSAM2_latest.pt
CFG=configs/sam2.1_hiera_t512.yaml          # prefisso configs/ obbligatorio (root Hydra = pacchetto sam2)
DATA=data/datasets/MRI_muscle/processed/WATER
EVAL_CFG=configs/mri_muscle.yaml
LABEL=6                                       # R_HS
DEV=cuda

echo "=== REFINE 0 (baseline) ==="
python3 scripts/eval/eval_medsam2.py \
  --config "$EVAL_CFG" \
  --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" \
  --test_label "$LABEL" --prompt_mode support_bbox \
  --refine_iters 0 \
  --device "$DEV" \
  --save_dir results/refine_ab/refine0 --save_topk 3

echo "=== REFINE 2 ==="
python3 scripts/eval/eval_medsam2.py \
  --config "$EVAL_CFG" \
  --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" \
  --test_label "$LABEL" --prompt_mode support_bbox \
  --refine_iters 2 \
  --device "$DEV" \
  --save_dir results/refine_ab/refine2 --save_topk 3

echo "=== DONE — confronta results/refine_ab/refine0/scores.csv vs refine2/scores.csv ==="
