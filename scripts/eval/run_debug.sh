#!/usr/bin/env bash
# MedSAM2 / MRI_muscle experiments + debug -- Linux server (CUDA).
#
#   ./scripts/eval/run_debug.sh <experiment>
#
#   all_muscles  eval support_bbox, 8 labels, query_slice=auto  -> results/all_muscles/
#   keyslice     same but query_slice=key (operator proxy)      -> results/all_muscles_keyslice/
#   refine_ab    A/B refine_iters 0 vs 2, label R_HS            -> results/refine_ab/
#   oracle       vis, box from query GT: isolates SAM2 itself   -> results/debug_vis/oracle_key/
#   support      vis, box from matching (R_SA, R_GR)            -> results/debug_vis/<label>_auto/
#   allsupp      vis, 1 R_SA query vs EVERY support (variance)  -> results/debug_vis/R_SA_HV010_allsupp/
#
# Triage any run:  python3 scripts/eval/debug_medsam2.py triage <scores.csv>
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI
export PYTHONPATH=.

CKPT=third_party/MedSAM2/checkpoints/MedSAM2_latest.pt
CFG=configs/sam2.1_hiera_t512.yaml   # configs/ prefix required (Hydra root = sam2 package)
DATA=data/datasets/MRI_muscle/processed/WATER
EVAL_CFG=configs/mri_muscle.yaml
DEV=cuda

EVAL="python3 scripts/eval/eval_medsam2.py --config $EVAL_CFG --medsam2_ckpt $CKPT
      --sam2_cfg $CFG --target_data_dir $DATA --device $DEV"
VIS="python3 scripts/eval/debug_medsam2.py vis --config $EVAL_CFG --medsam2_ckpt $CKPT
     --sam2_cfg $CFG --target_data_dir $DATA --device $DEV --refine_iters 1"

case "${1:-}" in

all_muscles)  # no --test_label => evaluate() runs labels 1..8
  $EVAL --prompt_mode support_bbox --refine_iters 1 \
        --save_dir results/all_muscles --save_topk 2
  echo "=== DONE — triage: python3 scripts/eval/debug_medsam2.py triage results/all_muscles/scores.csv"
  ;;

keyslice)     # start slice = max cross-section; box still 100% from similarity
  $EVAL --prompt_mode support_bbox --query_slice key --refine_iters 1 \
        --save_dir results/all_muscles_keyslice --save_topk 2
  echo "=== DONE — triage: python3 scripts/eval/debug_medsam2.py triage results/all_muscles_keyslice/scores.csv"
  ;;

refine_ab)    # same seed/pairing, only refine_iters changes
  for R in 0 2; do
    echo "=== REFINE $R ==="
    $EVAL --test_label 6 --prompt_mode support_bbox --refine_iters "$R" \
          --save_dir "results/refine_ab/refine$R" --save_topk 3
  done
  echo "=== DONE — compare results/refine_ab/refine{0,2}/scores.csv"
  ;;

oracle)       # box = query GT: MedSAM2 upper bound, no matching involved
  $VIS --box_source oracle --query_slice key --out_dir results/debug_vis/oracle_key
  echo "=== DONE — high Dice here means the bottleneck is the matching, not SAM2"
  ;;

support)      # the two thin muscles that fail (R_SA=7, R_GR=8)
  for L in 7 8; do
    NAME=$([ "$L" = 7 ] && echo R_SA || echo R_GR)
    $VIS --box_source support --test_labels "$L" --query_slice auto \
         --out_dir "results/debug_vis/${NAME}_auto"
  done
  echo "=== DONE — boxiou~0 in the filename = mislocation (Regime A)"
  ;;

allsupp)      # how much the result depends on WHICH support is drawn
  $VIS --box_source support --test_labels 7 --only HV010_1_stack2 --all_supports \
       --query_slice auto --out_dir results/debug_vis/R_SA_HV010_allsupp
  ;;

*)
  sed -n '2,12p' "$0"; exit 1
  ;;
esac
