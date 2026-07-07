#!/usr/bin/env bash
# Debug VISIVO bbox+similarita' (support_bbox) — SERVER CUDA.
# Salva PNG a 3 pannelli (support+GT | similarity+BOX+GT | BOX+GT+pred) per VEDERE
# dove il box predetto cade. box_gt_iou nel filename distingue mislocation (Regime A,
# box_gt_iou~0) da box-giusto-maschera-sbagliata (box_gt_iou alto, Dice basso).
# I Dice qui riproducono results/scores.csv (auto, refine=1): sanity check.
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI
export PYTHONPATH=.

CKPT=third_party/MedSAM2/checkpoints/MedSAM2_latest.pt
CFG=configs/sam2.1_hiera_t512.yaml
DATA=data/datasets/MRI_muscle/processed/WATER
EVAL_CFG=configs/mri_muscle.yaml
DEV=cuda

# 1) R_SA (label 7) — tutte le query, support come nella eval (auto). 20 PNG.
python3 scripts/eval/debug_support_bbox_vis.py \
  --config "$EVAL_CFG" --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" --test_label 7 \
  --query_slice auto --refine_iters 1 --device "$DEV" \
  --out_dir results/debug_vis/R_SA_auto

# 2) R_GR (label 8) — tutte le query.
python3 scripts/eval/debug_support_bbox_vis.py \
  --config "$EVAL_CFG" --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" --test_label 8 \
  --query_slice auto --refine_iters 1 --device "$DEV" \
  --out_dir results/debug_vis/R_GR_auto

# 3) Deep-dive VARIANZA: una query R_SA che fallisce, disegnata contro OGNI support.
#    Mostra quanto il risultato dipende da CHI e' il support (ipotesi similarita').
python3 scripts/eval/debug_support_bbox_vis.py \
  --config "$EVAL_CFG" --medsam2_ckpt "$CKPT" --sam2_cfg "$CFG" \
  --target_data_dir "$DATA" --test_label 7 \
  --only HV010_1_stack2 --all_supports \
  --query_slice auto --refine_iters 1 --device "$DEV" \
  --out_dir results/debug_vis/R_SA_HV010_allsupp

echo "=== DONE — guarda i PNG in results/debug_vis/ ==="
