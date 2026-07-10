#!/usr/bin/env bash
# MedSAM2 / MRI_muscle_2 experiments + debug -- Linux server (CUDA).
# Mirror of run_debug.sh, retargeted at the single-leg MRI_muscle_2 dataset.
# Labels here: BG=0 QF=1 HS=2 SA=3 GR=4 AD=5 GLUT=6 (see configs/mri_muscle_2.yaml).
#
#   ./scripts/eval/run_debug_mri2.sh <experiment>
#
#   all_muscles  eval support_bbox, 6 labels, query_slice=auto  -> results/mri_muscle_2/all_muscles/
#   keyslice     same but query_slice=key (operator proxy)      -> results/mri_muscle_2/all_muscles_keyslice/
#   refine_ab    A/B refine_iters 0 vs 2, label HS               -> results/mri_muscle_2/refine_ab/
#   oracle       vis, GT box on the key slice: isolates SAM2   -> results/mri_muscle_2/debug_vis/oracle_key/
#   oracle_perslice  vis, GT box on every slice: no propagation -> results/mri_muscle_2/debug_vis/oracle_perslice/
#   anchors      B4 sweep: 1/2/4/8 box anchors, all 6 labels     -> results/mri_muscle_2/anchors/n<N>/
#   bag          B1 vis: K support slices, SA+GR only            -> results/mri_muscle_2/debug_vis/bag_k<K>/
#   bag_key      B1 vis with slice frozen (query_slice=key)      -> results/mri_muscle_2/debug_vis/bag_key_k<K>/
#   bag_eval     B1 sweep: K=1/3/5, all 6 labels                 -> results/mri_muscle_2/bag/k<K>/
#   mc           B2 vis: cross-class competition, single-leg mode (--single_leg)
#                -> results/mri_muscle_2/debug_vis/mc_k3/
#   mc_nocc      same as mc, seed-only CC (pre-fix ablation), leg-crop stays on
#                -> results/mri_muscle_2/debug_vis/mc_k3_nocc/
#   support      vis, box from matching (SA, GR)                -> results/mri_muscle_2/debug_vis/<label>_auto/
#   allsupp      vis, 1 SA query vs EVERY support (variance)    -> results/mri_muscle_2/debug_vis/SA_allsupp/
#   dice [dir]   reprint the table of a past run; no dir => every run under results/mri_muscle_2/
#
# Each experiment writes scores.csv + dice_by_z.csv in its out dir and prints its own table;
# vis experiments also write one debug PNG per scan there.
set -euo pipefail

cd /home/utente/Scrivania/.Giuseppe/OpenFSS-MRI
export PYTHONPATH=.

CKPT=third_party/MedSAM2/checkpoints/MedSAM2_latest.pt
CFG=configs/sam2.1_hiera_t512.yaml   # configs/ prefix required (Hydra root = sam2 package)
DATA=data/datasets/MRI_muscle_2/processed/WATER
EVAL_CFG=configs/mri_muscle_2.yaml
DEV=cuda

EVAL="python3 scripts/eval/eval_medsam2.py --config $EVAL_CFG --medsam2_ckpt $CKPT
      --sam2_cfg $CFG --target_data_dir $DATA --device $DEV"
VIS="python3 scripts/eval/debug_medsam2.py vis --config $EVAL_CFG --medsam2_ckpt $CKPT
     --sam2_cfg $CFG --target_data_dir $DATA --device $DEV --refine_iters 1"
MCVIS="python3 scripts/eval/debug_medsam2.py mcvis --config $EVAL_CFG --medsam2_ckpt $CKPT
       --sam2_cfg $CFG --target_data_dir $DATA --device $DEV --refine_iters 1 --single_leg"
TRIAGE="python3 scripts/eval/debug_medsam2.py triage"

case "${1:-}" in

all_muscles)  # no --test_label => evaluate() runs labels 1..6
  OUT=results/mri_muscle_2/all_muscles
  $EVAL --prompt_mode support_bbox --refine_iters 1 --save_dir $OUT --save_topk 2
  $TRIAGE $OUT
  ;;

keyslice)     # start slice = max cross-section; box still 100% from similarity
  OUT=results/mri_muscle_2/all_muscles_keyslice
  $EVAL --prompt_mode support_bbox --query_slice key --refine_iters 1 \
        --save_dir $OUT --save_topk 2
  $TRIAGE $OUT
  ;;

refine_ab)    # same seed/pairing, only refine_iters changes (HS=2)
  for R in 0 2; do
    $EVAL --test_label 2 --prompt_mode support_bbox --refine_iters "$R" \
          --save_dir "results/mri_muscle_2/refine_ab/refine$R" --save_topk 3
  done
  for R in 0 2; do
    echo; echo "########## refine_iters=$R ##########"
    $TRIAGE "results/mri_muscle_2/refine_ab/refine$R"
  done
  ;;

oracle)       # box = query GT on ONE slice: MedSAM2 upper bound, no matching involved
  OUT=results/mri_muscle_2/debug_vis/oracle_key
  $VIS --box_source oracle --query_slice key --out_dir $OUT
  $TRIAGE $OUT
  echo "=== PNGs in $OUT/ — high Dice here means the bottleneck is the matching, not SAM2"
  ;;

oracle_perslice)  # box = query GT on EVERY slice: nothing left to propagate.
  OUT=results/mri_muscle_2/debug_vis/oracle_perslice   # gap vs `oracle` = cost of z-propagation alone
  $VIS --box_source oracle --query_slice auto --out_dir $OUT
  $TRIAGE $OUT
  ;;

anchors)      # B4: re-anchor the support box on N slices. n=1 must reproduce all_muscles.
  for N in 1 2 4 8; do
    $EVAL --prompt_mode support_bbox --refine_iters 1 --n_anchors "$N" \
          --save_dir "results/mri_muscle_2/anchors/n$N" --save_topk 1
  done
  for N in 1 2 4 8; do
    echo; echo "########## n_anchors=$N ##########"
    $TRIAGE "results/mri_muscle_2/anchors/n$N"
  done
  ;;

bag)          # B1: does a K-slice support bag sharpen the similarity map? Look at the PNGs
  for K in 1 3 5; do        # middle panel = score map; k1 must reproduce results/mri_muscle_2/debug_vis/*_auto
    OUT="results/mri_muscle_2/debug_vis/bag_k$K"
    $VIS --box_source support --test_labels 3 4 --query_slice auto \
         --support_slices "$K" --out_dir "$OUT"
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K ##########"
    $TRIAGE "results/mri_muscle_2/debug_vis/bag_k$K"
  done
  echo "=== compare the middle panel across bag_k1/k3/k5 for the same scan"
  ;;

bag_key)      # B1 with the slice FROZEN (query_slice=key): box always on the same slice for
  for K in 1 3 5; do        # every K, so the ONLY variable is the bag. Isolates box quality
    OUT="results/mri_muscle_2/debug_vis/bag_key_k$K"    # from slice selection (which bag also changes).
    $VIS --box_source support --test_labels 3 4 --query_slice key \
         --support_slices "$K" --out_dir "$OUT"
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K (slice frozen) ##########"
    $TRIAGE "results/mri_muscle_2/debug_vis/bag_key_k$K"
  done
  echo "=== boxiou rising with K here = B1 works, slice-selection is the separate problem"
  ;;

bag_eval)     # B1 full sweep, only worth running if `bag` shows a sharper map. k=1 == all_muscles
  for K in 1 3 5; do
    $EVAL --prompt_mode support_bbox --refine_iters 1 --support_slices "$K" \
          --save_dir "results/mri_muscle_2/bag/k$K" --save_topk 1
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K ##########"
    $TRIAGE "results/mri_muscle_2/bag/k$K"
  done
  ;;

mc)           # B2: cross-class competition, single-leg mode (whole body = one group, no L/R)
  OUT=results/mri_muscle_2/debug_vis/mc_k3    # same seed/pairing as bag_key_k3 -> boxiou compares directly
  $MCVIS --support_slices 3 --out_dir $OUT
  $TRIAGE $OUT
  echo; echo "########## reference: bag_key_k3 (binary bag, same slice) ##########"
  [ -d results/mri_muscle_2/debug_vis/bag_key_k3 ] && $TRIAGE results/mri_muscle_2/debug_vis/bag_key_k3
  ;;

mc_nocc)      # ablation: same as mc, but _box_from_blob reverted to pre-fix seed-only CC
              # (single-leg crop from fix v3 stays ON -- isolates the CC fix alone)
  OUT=results/mri_muscle_2/debug_vis/mc_k3_nocc
  $MCVIS --support_slices 3 --out_dir $OUT --cc_mode seed_only
  $TRIAGE $OUT
  echo; echo "########## reference: mc_k3 (dilate_largest, current fix) ##########"
  [ -d results/mri_muscle_2/debug_vis/mc_k3 ] && $TRIAGE results/mri_muscle_2/debug_vis/mc_k3
  ;;

support)      # the two thin muscles that fail (SA=3, GR=4)
  for L in 3 4; do
    NAME=$([ "$L" = 3 ] && echo SA || echo GR)
    OUT="results/mri_muscle_2/debug_vis/${NAME}_auto"
    $VIS --box_source support --test_labels "$L" --query_slice auto --out_dir "$OUT"
    $TRIAGE "$OUT"
    echo "=== PNGs in $OUT/ — boxiou~0 in the filename = mislocation (Regime A)"
  done
  ;;

allsupp)      # how much the result depends on WHICH support is drawn
  OUT=results/mri_muscle_2/debug_vis/SA_allsupp
  $VIS --box_source support --test_labels 3 --all_supports \
       --query_slice auto --out_dir $OUT
  $TRIAGE $OUT
  echo "=== PNGs in $OUT/ — one per candidate support"
  ;;

dice)         # reprint a past run: reads scores.csv, does not touch the model
  if [ -n "${2:-}" ]; then
    $TRIAGE "$2"
  else
    shopt -s globstar nullglob   # globstar: refine_ab nests scores.csv two levels down
    FOUND=(results/mri_muscle_2/**/scores.csv)
    if [ ${#FOUND[@]} -eq 0 ]; then
      echo "No scores.csv under results/mri_muscle_2/ -- run an experiment first"; exit 1
    fi
    for f in "${FOUND[@]}"; do
      echo; echo "########## ${f%/scores.csv} ##########"
      $TRIAGE "$f"
    done
  fi
  ;;

*)
  sed -n '2,22p' "$0"; exit 1
  ;;
esac
