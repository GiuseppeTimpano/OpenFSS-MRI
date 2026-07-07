#!/usr/bin/env bash
# Eval support_bbox su TUTTI gli 8 muscoli — OPERATOR-IN-THE-LOOP: la fetta query di
# partenza e' fissata alla max-cross-section (proxy della scelta del clinico); il box
# resta 100% da similarita' support (nessun GT box). refine_iters=1.
# SERVER Linux (CUDA). Scrive results/all_muscles_keyslice/{summary.csv,scores.csv,...}.
# Confronta con results/all_muscles (query_slice=auto) e con l'oracolo prompt_mode=key.
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI

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
  --query_slice key \
  --refine_iters 1 \
  --device "$DEV" \
  --save_dir results/all_muscles_keyslice --save_topk 2

echo "=== DONE — triage: python3 scripts/eval/analyze_scores.py results/all_muscles_keyslice/scores.csv ==="
