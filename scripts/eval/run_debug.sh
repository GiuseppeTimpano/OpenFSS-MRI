#!/usr/bin/env bash
# MedSAM2 / MRI_muscle experiments + debug -- Linux server (CUDA).
#
#   ./scripts/eval/run_debug.sh <experiment>
#
#   all_muscles  eval support_bbox, 8 labels, query_slice=auto  -> results/all_muscles/
#   keyslice     same but query_slice=key (operator proxy)      -> results/all_muscles_keyslice/
#   refine_ab    A/B refine_iters 0 vs 2, label R_HS            -> results/refine_ab/
#   oracle       vis, GT box on the key slice: isolates SAM2   -> results/debug_vis/oracle_key/
#   oracle_perslice  vis, GT box on every slice: no propagation -> results/debug_vis/oracle_perslice/
#   anchors      B4 sweep: 1/2/4/8 box anchors, all 8 labels     -> results/anchors/n<N>/
#   bag          B1 vis: K support slices, R_SA+R_GR only        -> results/debug_vis/bag_k<K>/
#   bag_key      B1 vis with slice frozen (query_slice=key)      -> results/debug_vis/bag_key_k<K>/
#   bag_eval     B1 sweep: K=1/3/5, all 8 labels                 -> results/bag/k<K>/
#   mc           B2 vis: cross-class competition, slice frozen    -> results/debug_vis/mc_k3/
#   support      vis, box from matching (R_SA, R_GR)            -> results/debug_vis/<label>_auto/
#   allsupp      vis, 1 R_SA query vs EVERY support (variance)  -> results/debug_vis/R_SA_HV010_allsupp/
#   dice [dir]   reprint the table of a past run; no dir => every run under results/
#
# Each experiment writes scores.csv + dice_by_z.csv in its out dir and prints its own table;
# vis experiments also write one debug PNG per scan there.
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
MCVIS="python3 scripts/eval/debug_medsam2.py mcvis --config $EVAL_CFG --medsam2_ckpt $CKPT
       --sam2_cfg $CFG --target_data_dir $DATA --device $DEV --refine_iters 1"
TRIAGE="python3 scripts/eval/debug_medsam2.py triage"

case "${1:-}" in

all_muscles)  # no --test_label => evaluate() runs labels 1..8
  OUT=results/all_muscles
  $EVAL --prompt_mode support_bbox --refine_iters 1 --save_dir $OUT --save_topk 2
  $TRIAGE $OUT
  ;;

keyslice)     # start slice = max cross-section; box still 100% from similarity
  OUT=results/all_muscles_keyslice
  $EVAL --prompt_mode support_bbox --query_slice key --refine_iters 1 \
        --save_dir $OUT --save_topk 2
  $TRIAGE $OUT
  ;;

refine_ab)    # same seed/pairing, only refine_iters changes
  for R in 0 2; do
    $EVAL --test_label 6 --prompt_mode support_bbox --refine_iters "$R" \
          --save_dir "results/refine_ab/refine$R" --save_topk 3
  done
  for R in 0 2; do
    echo; echo "########## refine_iters=$R ##########"
    $TRIAGE "results/refine_ab/refine$R"
  done
  ;;

oracle)       # box = query GT on ONE slice: MedSAM2 upper bound, no matching involved
  OUT=results/debug_vis/oracle_key
  $VIS --box_source oracle --query_slice key --out_dir $OUT
  $TRIAGE $OUT
  echo "=== PNGs in $OUT/ — high Dice here means the bottleneck is the matching, not SAM2"
  ;;

oracle_perslice)  # box = query GT on EVERY slice: nothing left to propagate.
  OUT=results/debug_vis/oracle_perslice   # gap vs `oracle` = cost of z-propagation alone
  $VIS --box_source oracle --query_slice auto --out_dir $OUT
  $TRIAGE $OUT
  ;;

anchors)      # B4: re-anchor the support box on N slices. n=1 must reproduce all_muscles.
  for N in 1 2 4 8; do
    $EVAL --prompt_mode support_bbox --refine_iters 1 --n_anchors "$N" \
          --save_dir "results/anchors/n$N" --save_topk 1
  done
  for N in 1 2 4 8; do
    echo; echo "########## n_anchors=$N ##########"
    $TRIAGE "results/anchors/n$N"
  done
  ;;

bag)          # B1: does a K-slice support bag sharpen the similarity map? Look at the PNGs
  for K in 1 3 5; do        # middle panel = score map; k1 must reproduce results/debug_vis/*_auto
    OUT="results/debug_vis/bag_k$K"
    $VIS --box_source support --test_labels 7 8 --query_slice auto \
         --support_slices "$K" --out_dir "$OUT"
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K ##########"
    $TRIAGE "results/debug_vis/bag_k$K"
  done
  echo "=== compare the middle panel across bag_k1/k3/k5 for the same scan"
  ;;

bag_key)      # B1 with the slice FROZEN (query_slice=key): box always on the same slice for
  for K in 1 3 5; do        # every K, so the ONLY variable is the bag. Isolates box quality
    OUT="results/debug_vis/bag_key_k$K"    # from slice selection (which bag also changes).
    $VIS --box_source support --test_labels 7 8 --query_slice key \
         --support_slices "$K" --out_dir "$OUT"
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K (slice frozen) ##########"
    $TRIAGE "results/debug_vis/bag_key_k$K"
  done
  echo "=== boxiou rising with K here = B1 works, slice-selection is the separate problem"
  ;;

bag_eval)     # B1 full sweep, only worth running if `bag` shows a sharper map. k=1 == all_muscles
  for K in 1 3 5; do
    $EVAL --prompt_mode support_bbox --refine_iters 1 --support_slices "$K" \
          --save_dir "results/bag/k$K" --save_topk 1
  done
  for K in 1 3 5; do
    echo; echo "########## support_slices=$K ##########"
    $TRIAGE "results/bag/k$K"
  done
  ;;

mc)           # B2: cross-class competition instead of the binary pos/neg bag. Slice frozen,
  OUT=results/debug_vis/mc_k3    # same seed/pairing as bag_key_k3 -> boxiou compares directly
  $MCVIS --support_slices 3 --out_dir $OUT
  $TRIAGE $OUT
  echo; echo "########## reference: bag_key_k3 (binary bag, same slice) ##########"
  [ -d results/debug_vis/bag_key_k3 ] && $TRIAGE results/debug_vis/bag_key_k3
  echo "=== SA/GR boxiou up and QF/HS not collapsing = B2 works, wire it into eval_medsam2.py"
  ;;

support)      # the two thin muscles that fail (R_SA=7, R_GR=8)
  for L in 7 8; do
    NAME=$([ "$L" = 7 ] && echo R_SA || echo R_GR)
    OUT="results/debug_vis/${NAME}_auto"
    $VIS --box_source support --test_labels "$L" --query_slice auto --out_dir "$OUT"
    $TRIAGE "$OUT"
    echo "=== PNGs in $OUT/ — boxiou~0 in the filename = mislocation (Regime A)"
  done
  ;;

allsupp)      # how much the result depends on WHICH support is drawn
  OUT=results/debug_vis/R_SA_HV010_allsupp
  $VIS --box_source support --test_labels 7 --only HV010_1_stack2 --all_supports \
       --query_slice auto --out_dir $OUT
  $TRIAGE $OUT
  echo "=== PNGs in $OUT/ — one per candidate support"
  ;;

dice)         # reprint a past run: reads scores.csv, does not touch the model
  if [ -n "${2:-}" ]; then
    $TRIAGE "$2"
  else
    shopt -s globstar nullglob   # globstar: refine_ab nests scores.csv two levels down
    FOUND=(results/**/scores.csv)
    if [ ${#FOUND[@]} -eq 0 ]; then
      echo "No scores.csv under results/ -- run an experiment first"; exit 1
    fi
    for f in "${FOUND[@]}"; do
      echo; echo "########## ${f%/scores.csv} ##########"
      $TRIAGE "$f"
    done
  fi
  ;;

*)
  sed -n '2,19p' "$0"; exit 1
  ;;
esac
