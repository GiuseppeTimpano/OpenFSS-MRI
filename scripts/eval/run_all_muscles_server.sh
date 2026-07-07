#!/usr/bin/env bash
# Eval support_bbox su TUTTI gli 8 muscoli (label 1..8) — MRI_muscle, refine_iters=1.
# SERVER Linux (CUDA). Scrive results/all_muscles/{summary.csv, scores.csv, best/worst .nii.gz}.
# Nessun --test_label => evaluate() gira su range(1, len(label_names)) = 1..8.
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI
source .venv/bin/activate

export PYTHONPATH=.

CKPT=third_party/MedSAM2/checkpoints/MedSAM2_latest.pt
CFG=configs/sam2.1_hiera_t512.yaml          # prefisso configs/ obbligatorio (root Hydra = pacchetto sam2)
DATA=data/datasets/MRI_muscle/processed/WATER
EVAL_CFG=configs/mri_muscle.yaml
DEV=cuda

python3 scripts/eval/eval_medsam2.py \
  --config "$EVAL_CFG" \
  --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" \
  --prompt_mode support_bbox \
  --refine_iters 1 \
  --device "$DEV" \
  --save_dir results/all_muscles --save_topk 2

echo "=== DONE — triage: python3 scripts/eval/analyze_scores.py results/all_muscles/scores.csv ==="
