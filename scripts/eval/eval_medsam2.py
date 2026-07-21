"""
MedSAM2 zero-shot eval (P1) -- volume-level, oracle box + propagation. Promptable,
not few-shot, so separate harness from scripts/prototype/test.py; reuses
eval_common.Scores for the shared per-organ + MEAN table.

Per organ: ORACLE box from GT (upper bound, NOT deployable -- deployable variant
is P3) on the prompt slice(s), propagated both directions, scored on FG slices.

prompt_mode: perslice (default, oracle box every FG slice) | key (one box on
largest-area slice + propagation, MedSAM2's native usage) | support_bbox (box
prompt derived from a random support scan's mask, PerSAM-style SAM2-embedding
matching, no query GT read -- dense bag-of-vectors + body mask + similarity-blob
bbox, see models/support_prompt.py:support_anchors_dense_bodymasked_bbox) |
support_multiclass (B2: same support-derived box, but every muscle TYPE gets its
own bag and they compete cell-by-cell on the query's key slice, see
models/support_prompt.py:build_multiclass_bags/multiclass_boxes_for_frame -- this
is the full propagation + aggregated-Dice wiring of the technique previously only
available as a single-frame diagnostic in debug_medsam2.py cmd_mcvis) |
support_multiclass_mask (mask-prompt ablation of support_multiclass: same bag/
winner-take-all scoring, but the raw similarity blob is fed to SAM2 as a mask prompt
instead of being reduced to a box, see models/support_prompt.py:multiclass_masks_for_frame,
multiclass_mask_anchors and models/medsam2_adapter.py:segment_volume_mask). --n_anchors > 1
re-anchors that box/mask on several slices, so propagation restarts before it decays
(support_bbox: N best-scoring query slices; support_multiclass_mask: N query FG slices
where the class survives the rival winner-take-all, which also rescues the catastrophic-
zero case where the class loses on the single frozen key slice). --support_slices > 1
(B1) builds the Pos/Neg (or multiclass) bag from several slices of the same support
volume instead of its key slice alone -- still 1-shot, richer bag. --split_legs
(support_multiclass, support_multiclass_mask; bilateral datasets only) crops query
+ support to one leg (models/support_prompt.py:leg_crop_boxes) and reuses the
existing single_leg=True pipeline unmodified on each side independently: a
bilateral scan crams both legs into the same fixed 512x512 SAM2 embedding grid a
single-leg scan gives entirely to one leg (~1.6x less effective per-leg resolution
on MRI_muscle vs MRI_muscle_2), which starves small/thin muscles (SA, GR)
disproportionately -- the winner-take-all itself was already 4-way, not 8-way
(support_bag_slices already pools L+R into one bag per type), so this is a
resolution fix, not a competition fix. --anchor_coherence FRAC (support_multiclass_mask
only, requires --n_anchors > 1 to matter) filters n_anchors candidate slices by centroid
distance to a reference (key slice if the class survives there, else the median centroid
across candidates) before ranking them by score, and clips the scattered-blob mask
fallback the same way (see models/support_prompt.py:multiclass_mask_anchors /
_mask_from_blob docstrings) -- guards against a spatially wrong but high-scoring blob on
some other slice (common for small/weak-texture classes like GR, SA) becoming an anchor
and corrupting the whole-volume SAM2 propagation. None (default) = unchanged behaviour.

Normalization is MedSAM2's (uint8+512+ImageNet), not the baseline's z-score --
see models/medsam2_adapter.py.
"""
import argparse
import csv
import glob
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from eval_common import Scores, aggregate_and_print
from models.medsam2_adapter import MedSAM2Segmenter, volume_to_uint8
from models.support_prompt import (build_multiclass_bags, key_slice, leg_crop_boxes,
                                   left_is_low_x, multiclass_boxes_for_frame,
                                   multiclass_mask_anchors, multiclass_masks_for_frame,
                                   muscle_types, muscle_types_single, pick_support_slices,
                                   support_anchors_dense_bodymasked_bbox,
                                   support_bag_slices, support_bag_slices_single)


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_raw(data_dir: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    """Load RAW image (no z-score) + int label. MedSAM2 does its own [0,255] scaling."""
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    return img, lbl


def _bbox(mask2d: np.ndarray, margin: int, H: int, W: int) -> np.ndarray:
    ys, xs = np.where(mask2d)
    x0 = max(0, int(xs.min()) - margin); x1 = min(W - 1, int(xs.max()) + margin)
    y0 = max(0, int(ys.min()) - margin); y1 = min(H - 1, int(ys.max()) + margin)
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def _build_boxes(fg_mask: np.ndarray, fg_idx: np.ndarray, z0: int,
                 mode: str, margin: int) -> dict[int, np.ndarray]:
    """{local_frame_idx -> oracle box}. Local idx = z - z0 (crop offset)."""
    H, W = fg_mask.shape[1], fg_mask.shape[2]
    if mode == 'key':
        areas = fg_mask[fg_idx].reshape(len(fg_idx), -1).sum(1)
        zc = int(fg_idx[int(np.argmax(areas))])
        return {zc - z0: _bbox(fg_mask[zc], margin, H, W)}
    return {int(z) - z0: _bbox(fg_mask[z], margin, H, W) for z in fg_idx}


def _leg_crop_bag(seg, query_data_dir: str, supp_sid: str, vol_u8: np.ndarray,
                  types: dict, label_name: str, support_slices: int, support_min_gap: int,
                  split_cache: dict):
    """--split_legs orchestration shared by support_multiclass / support_multiclass_mask.
    Crops query + support to ONE leg (side read off label_name's 'L_'/'R_' prefix) and
    hands back everything needed to call the existing single_leg=True pipeline
    unmodified on the crop -- see models/support_prompt.py:leg_crop_boxes for why
    (per-leg resolution, not the winner-take-all competition, is what the bilateral
    path was losing to single-leg).

    split_cache: {(supp_sid, side) -> (bags, left_is_low_x) | None}, shared across
    classes/scans within one evaluate() call so the same support crop's bags aren't
    rebuilt (re-embedded) for every query scan that happens to draw the same support.

    Returns (bags_leg, vol_u8_leg, (y0, y1, x0, x1), mtype), or None when either the
    support or this query volume has no detectable leg on that side (crop degenerates
    -- caller falls back to an empty prediction, same as the existing NO BOX/NO MASK
    lost-to-a-rival case).
    """
    side, mtype = label_name.split('_', 1)

    if (supp_sid, side) not in split_cache:
        supp_img, supp_lbl = _load_raw(query_data_dir, supp_sid)
        supp_vol_u8 = volume_to_uint8(supp_img)
        supp_low_x = left_is_low_x(supp_lbl, types)
        supp_box = leg_crop_boxes(supp_vol_u8, supp_low_x).get(side)
        if supp_box is None:
            split_cache[(supp_sid, side)] = None
        else:
            sy0, sy1, sx0, sx1 = supp_box
            supp_vol_leg = supp_vol_u8[:, sy0:sy1, sx0:sx1]
            supp_lbl_leg = supp_lbl[:, sy0:sy1, sx0:sx1]
            types_leg = {t: v[side] for t, v in types.items()}
            supp_slices, _ = support_bag_slices_single(
                supp_vol_leg, supp_lbl_leg, types_leg, support_slices, support_min_gap)
            bags_leg = build_multiclass_bags(seg, supp_slices)
            split_cache[(supp_sid, side)] = (bags_leg, supp_low_x)

    cached = split_cache[(supp_sid, side)]
    if cached is None:
        return None
    bags_leg, low_x = cached

    q_box = leg_crop_boxes(vol_u8, low_x).get(side)
    if q_box is None:
        return None
    y0, y1, x0, x1 = q_box
    vol_u8_leg = vol_u8[:, y0:y1, x0:x1]
    return bags_leg, vol_u8_leg, (y0, y1, x0, x1), mtype


def evaluate(cfg: dict, checkpoint: str, model_cfg: str,
             target_data_dir: str | None, fold: int | None,
             eval_labels: list[int] | None, prompt_mode: str, margin: int,
             device: str, save_dir: str | None, save_topk: int = 1,
             seed: int = 42, refine_iters: int = 0,
             query_slice: str = 'auto', n_anchors: int = 1,
             anchor_min_gap: int = 3, support_slices: int = 1,
             support_min_gap: int = 3, single_leg: bool = False,
             cc_mode: str = 'dilate_largest', neg_points: bool = False,
             max_neg_points: int = 3, split_legs: bool = False,
             anchor_coherence: float | None = None) -> dict:
    data_cfg    = cfg['data']
    data_dir    = data_cfg['data_dir']
    n_folds     = data_cfg['n_folds']
    label_names = data_cfg['label_names']
    if eval_labels is None:
        eval_labels = list(range(1, len(label_names)))

    types = None
    if prompt_mode in ('support_multiclass', 'support_multiclass_mask'):
        types = muscle_types_single(label_names) if single_leg else muscle_types(label_names)
        # {supp_sid -> (bags, left_is_low_x)}; spans every class since the multiclass bag
        # already covers every type -- only rebuilt when a class's random pairing draws a
        # support scan not seen yet for ANY class.
        mc_bag_cache: dict = {}
        # --split_legs: {(supp_sid, side) -> (bags, left_is_low_x) | None}, see _leg_crop_bag.
        split_cache: dict = {}

    query_data_dir = target_data_dir or data_dir
    if target_data_dir:
        paths = sorted(glob.glob(os.path.join(target_data_dir, 'image_*.nii.gz')))
        query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                      for p in paths]
    else:
        _, query_sids = get_fold_ids(data_dir, fold if fold is not None else 0, n_folds)
    # P* = pathological patients; exclude from query AND support pool (support drawn
    # from query_sids too) -- too different from HV support/query to trust matching.
    query_sids = [s for s in query_sids if not s.startswith('P')]
    if not query_sids:
        raise ValueError(f'No query scans found in {query_data_dir}')

    print(f'MedSAM2 zero-shot | queries={len(query_sids)} | prompt={prompt_mode} '
          f'| margin={margin} | eval_labels={eval_labels}')
    print(f'query dir: {query_data_dir}')

    seg = MedSAM2Segmenter(checkpoint, model_cfg, device=device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    class_dice: dict[str, float] = {}
    class_iou:  dict[str, float] = {}
    csv_rows: list[dict] = []

    for label_val in eval_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)
        print(f'\n== Class: {label_name} (label={label_val}) ==')
        scores = Scores()
        scan_ids: list[str] = []
        # kept only for this class's loop, discarded once best/worst are written out
        cls_img: dict[str, np.ndarray] = {}
        cls_gt:  dict[str, np.ndarray] = {}
        cls_pred: dict[str, np.ndarray] = {}

        support_fg_idx: dict[str, np.ndarray] = {}
        rng = None
        if prompt_mode in ('support_bbox', 'support_multiclass', 'support_multiclass_mask'):
            rng = random.Random(seed + label_val)
            for sid in query_sids:
                lbl = _read_nii(os.path.join(query_data_dir, f'label_{sid}.nii.gz'))
                idx = np.where((lbl == label_val).any(axis=(1, 2)))[0]
                if len(idx):
                    support_fg_idx[sid] = idx

        for qsid in query_sids:
            q_img, q_lbl = _load_raw(query_data_dir, qsid)
            q_fg = (q_lbl == label_val).astype(np.uint8)

            fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
            if len(fg_idx) == 0:
                print(f'  [SKIP] query {qsid} has no FG for {label_name}')
                continue

            z0, z1 = int(fg_idx.min()), int(fg_idx.max())
            vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]          # window full vol, then crop

            if prompt_mode == 'support_bbox':
                pool = [s for s in support_fg_idx if s != qsid]
                if not pool:
                    print(f'  [SKIP] query {qsid} has no support scan available for {label_name}')
                    continue
                supp_sid = rng.choice(pool)
                supp_img, supp_lbl = _load_raw(query_data_dir, supp_sid)
                supp_fg = (supp_lbl == label_val).astype(np.uint8)
                supp_vol_u8 = volume_to_uint8(supp_img)
                supp_zs = pick_support_slices(supp_fg, support_slices, support_min_gap)
                supp = [(supp_vol_u8[z], supp_fg[z].astype(bool)) for z in supp_zs]

                if query_slice == 'key':
                    # Operator-in-the-loop proxy: fix the start slice to the query's
                    # max-cross-section slice (a clinician would pick the reference slice),
                    # so the similarity picks the box ON THAT slice instead of also choosing
                    # the slice. Box is still 100% from support similarity -- NO query GT box
                    # is read (key_slice reads GT only to locate the slice, standing in for
                    # the human). Isolates box-quality-given-slice: compare vs 'auto' (adds
                    # slice-selection error) and vs the oracle prompt_mode=key (same slice,
                    # GT box) upper bound.
                    zc = key_slice(q_fg)
                    query_frames = [(zc - z0, vol_u8[zc - z0])]
                else:
                    query_frames = [(int(z) - z0, vol_u8[int(z) - z0]) for z in fg_idx]
                boxes = support_anchors_dense_bodymasked_bbox(
                    seg, supp, query_frames,
                    n_anchors=n_anchors, min_gap=anchor_min_gap)
                seg_crop = seg.segment_volume(vol_u8, boxes, refine_iters=refine_iters)
            elif prompt_mode == 'support_multiclass':
                pool = [s for s in support_fg_idx if s != qsid]
                if not pool:
                    print(f'  [SKIP] query {qsid} has no support scan available for {label_name}')
                    continue
                supp_sid = rng.choice(pool)

                if split_legs and not single_leg and '_' in label_name:
                    leg = _leg_crop_bag(seg, query_data_dir, supp_sid, vol_u8, types,
                                        label_name, support_slices, support_min_gap,
                                        split_cache)
                    if leg is None:
                        print(f'  [NO LEG] {qsid}: could not isolate the '
                              f'{label_name.split("_", 1)[0]} leg')
                        seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                    else:
                        bags_leg, vol_u8_leg, (y0, y1, x0, x1), mtype = leg
                        frame_idx = key_slice(q_fg) - z0
                        boxes_by_name, _ = multiclass_boxes_for_frame(
                            seg, bags_leg, vol_u8_leg[frame_idx], None,
                            single_leg=True, cc_mode=cc_mode,
                            neg_points=neg_points, max_neg_points=max_neg_points)
                        if mtype not in boxes_by_name:
                            print(f'  [NO BOX] {qsid}: {label_name} lost to a rival on '
                                  f'the key slice (split_legs)')
                            seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                        else:
                            npts = None
                            if neg_points:
                                _, box, pts = boxes_by_name[mtype]
                                if pts:
                                    npts = {frame_idx: pts}
                            else:
                                _, box = boxes_by_name[mtype]
                            seg_crop_leg = seg.segment_volume(
                                vol_u8_leg, {frame_idx: np.asarray(box, dtype=np.float32)},
                                refine_iters=refine_iters, neg_points=npts)
                            seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                            seg_crop[:, y0:y1, x0:x1] = seg_crop_leg
                else:
                    if supp_sid not in mc_bag_cache:
                        supp_img, supp_lbl = _load_raw(query_data_dir, supp_sid)
                        supp_vol_u8 = volume_to_uint8(supp_img)
                        bag_fn = support_bag_slices_single if single_leg else support_bag_slices
                        supp_slices, _ = bag_fn(supp_vol_u8, supp_lbl, types,
                                                support_slices, support_min_gap)
                        low_x = None if single_leg else left_is_low_x(supp_lbl, types)
                        bags = build_multiclass_bags(seg, supp_slices)
                        mc_bag_cache[supp_sid] = (bags, low_x)
                    bags, low_x = mc_bag_cache[supp_sid]

                    # frozen key-slice seed (operator-in-the-loop proxy, same as
                    # debug_medsam2.py cmd_mcvis) -- multiclass_boxes_for_frame scores one
                    # frame at a time, so there is no query_slice=auto sweep here yet.
                    frame_idx = key_slice(q_fg) - z0   # local index into the cropped vol_u8
                    boxes_by_name, _ = multiclass_boxes_for_frame(
                        seg, bags, vol_u8[frame_idx], low_x,
                        single_leg=single_leg, cc_mode=cc_mode,
                        neg_points=neg_points, max_neg_points=max_neg_points)
                    if label_name not in boxes_by_name:
                        print(f'  [NO BOX] {qsid}: {label_name} lost to a rival on the key slice')
                        seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                    else:
                        npts = None
                        if neg_points:
                            _, box, pts = boxes_by_name[label_name]
                            if pts:
                                npts = {frame_idx: pts}
                        else:
                            _, box = boxes_by_name[label_name]
                        seg_crop = seg.segment_volume(
                            vol_u8, {frame_idx: np.asarray(box, dtype=np.float32)},
                            refine_iters=refine_iters, neg_points=npts)
            elif prompt_mode == 'support_multiclass_mask':
                # Mask-prompt sibling of support_multiclass: same support bag / winner-take-all
                # scoring, but the raw similarity blob is fed to SAM2 as a mask prompt instead
                # of being reduced to an axis-aligned box (models/support_prompt.py
                # multiclass_masks_for_frame, models/medsam2_adapter.py segment_volume_mask).
                pool = [s for s in support_fg_idx if s != qsid]
                if not pool:
                    print(f'  [SKIP] query {qsid} has no support scan available for {label_name}')
                    continue
                supp_sid = rng.choice(pool)

                if split_legs and not single_leg and '_' in label_name:
                    leg = _leg_crop_bag(seg, query_data_dir, supp_sid, vol_u8, types,
                                        label_name, support_slices, support_min_gap,
                                        split_cache)
                    if leg is None:
                        print(f'  [NO LEG] {qsid}: could not isolate the '
                              f'{label_name.split("_", 1)[0]} leg')
                        seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                    else:
                        bags_leg, vol_u8_leg, (y0, y1, x0, x1), mtype = leg
                        zc = key_slice(q_fg) - z0
                        if n_anchors <= 1:
                            cand_frames = [(zc, vol_u8_leg[zc])]
                        else:
                            cand_frames = [(int(z) - z0, vol_u8_leg[int(z) - z0])
                                           for z in fg_idx]
                        mask_anchors = multiclass_mask_anchors(
                            seg, bags_leg, cand_frames, mtype, None,
                            n_anchors=n_anchors, min_gap=anchor_min_gap,
                            single_leg=True, cc_mode=cc_mode,
                            coherence_frac=anchor_coherence, key_fidx=zc)
                        if not mask_anchors:
                            print(f'  [NO MASK] {qsid}: {label_name} lost to every rival '
                                  f'candidate (split_legs)')
                            seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                        else:
                            seg_crop_leg = seg.segment_volume_mask(vol_u8_leg, mask_anchors)
                            seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                            seg_crop[:, y0:y1, x0:x1] = seg_crop_leg
                else:
                    if supp_sid not in mc_bag_cache:
                        supp_img, supp_lbl = _load_raw(query_data_dir, supp_sid)
                        supp_vol_u8 = volume_to_uint8(supp_img)
                        bag_fn = support_bag_slices_single if single_leg else support_bag_slices
                        supp_slices, _ = bag_fn(supp_vol_u8, supp_lbl, types,
                                                support_slices, support_min_gap)
                        low_x = None if single_leg else left_is_low_x(supp_lbl, types)
                        bags = build_multiclass_bags(seg, supp_slices)
                        mc_bag_cache[supp_sid] = (bags, low_x)
                    bags, low_x = mc_bag_cache[supp_sid]

                    # n_anchors=1 (default): single candidate = key slice, same lookup as
                    # before -- byte-identical output, no extra encoder passes. n_anchors>1:
                    # search every FG slice of the query for ones where the class survives
                    # the winner-take-all, prompt the N best (see multiclass_mask_anchors).
                    zc = key_slice(q_fg) - z0
                    if n_anchors <= 1:
                        cand_frames = [(zc, vol_u8[zc])]
                    else:
                        cand_frames = [(int(z) - z0, vol_u8[int(z) - z0]) for z in fg_idx]
                    mask_anchors = multiclass_mask_anchors(
                        seg, bags, cand_frames, label_name, low_x,
                        n_anchors=n_anchors, min_gap=anchor_min_gap,
                        single_leg=single_leg, cc_mode=cc_mode,
                        coherence_frac=anchor_coherence, key_fidx=zc)
                    if not mask_anchors:
                        print(f'  [NO MASK] {qsid}: {label_name} lost to every rival candidate')
                        seg_crop = np.zeros_like(vol_u8, dtype=q_fg.dtype)
                    else:
                        seg_crop = seg.segment_volume_mask(vol_u8, mask_anchors)
            else:
                boxes = _build_boxes(q_fg, fg_idx, z0, prompt_mode, margin)
                seg_crop = seg.segment_volume(vol_u8, boxes)        # [z1-z0+1,H,W]
            pred_full = np.zeros_like(q_fg)
            pred_full[z0:z1 + 1] = seg_crop

            pred_fg = torch.from_numpy(pred_full[fg_idx].astype(np.int64))
            gt_fg   = torch.from_numpy(q_fg[fg_idx].astype(np.int64))
            scores.record(pred_fg, gt_fg)
            scan_ids.append(qsid)
            print(f'  scan {qsid}: Dice={scores.patient_dice[-1]:.4f}  '
                  f'IoU={scores.patient_iou[-1]:.4f}')

            if save_dir and save_topk > 0:
                cls_img[qsid]  = q_img.astype(np.float32)
                cls_gt[qsid]   = q_fg
                cls_pred[qsid] = pred_full

        if scores.patient_dice:
            class_dice[label_name] = float(np.mean(scores.patient_dice))
            class_iou[label_name]  = float(np.mean(scores.patient_iou))
            print(f'  mean Dice={class_dice[label_name]:.4f}  '
                  f'mean IoU={class_iou[label_name]:.4f}')

            for sid, d, i in zip(scan_ids, scores.patient_dice, scores.patient_iou):
                csv_rows.append({'class': label_name, 'label': label_val,
                                  'scan': sid, 'dice': d, 'iou': i})

            if save_dir and save_topk > 0 and scan_ids:
                order = sorted(range(len(scan_ids)), key=lambda k: scores.patient_dice[k])
                worst_idx = set(order[:save_topk])
                best_idx  = set(order[-save_topk:])
                for k in worst_idx | best_idx:
                    sid = scan_ids[k]
                    tag = 'best' if k in best_idx else 'worst'
                    d = scores.patient_dice[k]
                    base = f'{label_name}_{tag}_{sid}_dice{d:.3f}'
                    sitk.WriteImage(sitk.GetImageFromArray(cls_img[sid]),
                                    os.path.join(save_dir, f'{base}_image.nii.gz'), True)
                    sitk.WriteImage(sitk.GetImageFromArray(cls_gt[sid]),
                                    os.path.join(save_dir, f'{base}_gt.nii.gz'), True)
                    sitk.WriteImage(sitk.GetImageFromArray(cls_pred[sid]),
                                    os.path.join(save_dir, f'{base}_pred.nii.gz'), True)

    if save_dir and csv_rows:
        csv_path = os.path.join(save_dir, 'scores.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'label', 'scan', 'dice', 'iou'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f'\nPer-scan scores written to {csv_path}')

    results = aggregate_and_print(class_dice, class_iou)

    if save_dir and results:
        summary_path = os.path.join(save_dir, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'dice', 'iou'])
            writer.writeheader()
            for name, vals in results.items():
                writer.writerow({'class': name, 'dice': vals['dice'], 'iou': vals['iou']})
        print(f'Summary written to {summary_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str, default='configs/resnet.yaml')
    parser.add_argument('--medsam2_ckpt',    type=str, required=True,
                        help='MedSAM2 checkpoint (.pt)')
    parser.add_argument('--sam2_cfg',        type=str, required=True,
                        help='SAM2.1 model config (e.g. sam2.1_hiera_t512.yaml)')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='query dir (processed image_*/label_*); default = config data_dir')
    parser.add_argument('--fold',            type=int, default=None,
                        help='only used when --target_data_dir is not given')
    parser.add_argument('--test_label',      type=int, nargs='+', default=None)
    parser.add_argument('--prompt_mode',     type=str, default='perslice',
                        choices=['perslice', 'key', 'support_bbox', 'support_multiclass',
                                 'support_multiclass_mask'])
    parser.add_argument('--margin',          type=int, default=3,
                        help='oracle box margin in pixels')
    parser.add_argument('--seed',            type=int, default=42,
                        help='random support-scan selection (prompt_mode=support_bbox)')
    parser.add_argument('--refine_iters',    type=int, default=0,
                        help='cascaded box->mask->box refinement on the seed frame before '
                             'propagation (prompt_mode=support_bbox); 0 = off')
    parser.add_argument('--query_slice',     type=str, default='auto',
                        choices=['auto', 'key'],
                        help='(prompt_mode=support_bbox) which query slice seeds the box: '
                             'auto = similarity picks the best FG slice; key = operator '
                             'proxy, fix to the query max-cross-section slice (box still '
                             'from similarity, no GT box read)')
    parser.add_argument('--n_anchors',       type=int, default=1,
                        help='(prompt_mode=support_bbox, query_slice=auto) prompt the box on '
                             'the N best-scoring slices instead of 1; 1 = previous behavior. '
                             '(prompt_mode=support_multiclass_mask) same idea for the mask '
                             'prompt: search every query FG slice instead of the single key '
                             'slice, prompt the N where the class survives the rival '
                             'winner-take-all; 1 = previous single-key-slice behavior')
    parser.add_argument('--anchor_min_gap',  type=int, default=3,
                        help='min z-distance between anchors (--n_anchors > 1)')
    parser.add_argument('--support_slices',  type=int, default=1,
                        help='(prompt_mode=support_bbox) B1: build the Pos/Neg bag from the K '
                             'best support slices instead of the key slice alone; 1 = previous '
                             'behavior. Same support volume, so still 1-shot')
    parser.add_argument('--support_min_gap', type=int, default=3,
                        help='min z-distance between support slices (--support_slices > 1)')
    parser.add_argument('--single_leg',      action='store_true',
                        help='(prompt_mode=support_multiclass) dataset has one leg per '
                             'volume (no L/R): label_names are bare type names, no side '
                             'split of the body mask')
    parser.add_argument('--cc_mode',         type=str, default='dilate_largest',
                        choices=['dilate_largest', 'union', 'seed_only'],
                        help='(prompt_mode=support_multiclass) _box_from_blob CC selection: '
                             'dilate_largest = current fix, union = superseded first fix, '
                             'seed_only = pre-fix ablation baseline')
    parser.add_argument('--neg_points',      action='store_true',
                        help='(prompt_mode=support_multiclass) box+neg-points hybrid: add '
                             'negative clicks on rival-won cells inside the box, to push '
                             'SAM2 off neighboring muscle an elongated box also covers '
                             '(e.g. soleus). Off by default (box-only, previous behavior)')
    parser.add_argument('--max_neg_points',  type=int, default=3,
                        help='(--neg_points) cap on negative clicks per box')
    parser.add_argument('--split_legs',      action='store_true',
                        help='(prompt_mode=support_multiclass, support_multiclass_mask; '
                             'bilateral datasets only, ignored with --single_leg) crop query '
                             '+ support to one leg and run the single-leg pipeline on each '
                             'side independently instead of competing both legs in the same '
                             'fixed 512x512 SAM2 grid -- fixes a resolution deficit (~1.6x '
                             'less effective per-leg pixels than a true single-leg scan) that '
                             'starves small/thin muscles (SA, GR). See '
                             'models/support_prompt.py:leg_crop_boxes')
    parser.add_argument('--anchor_coherence', type=float, default=None,
                        help='(prompt_mode=support_multiclass_mask, --n_anchors > 1) reject '
                             'candidate anchor slices whose winning blob sits farther than '
                             'FRAC * frame_diagonal from a reference centroid (key slice if '
                             'the class survives there, else the median centroid across '
                             'candidates), and clip the scattered-blob mask fallback the same '
                             'way. Guards against a spatially wrong but high-scoring blob on '
                             'another slice corrupting the whole-volume SAM2 propagation -- '
                             'see models/support_prompt.py:multiclass_mask_anchors docstring. '
                             'None (default) = off, unchanged behavior. Try e.g. 0.35')
    parser.add_argument('--save_dir',        type=str, default=None,
                        help='where to write scores.csv/summary.csv and best/worst volumes')
    parser.add_argument('--save_topk',       type=int, default=1,
                        help='per class, save nii.gz for N best + N worst scans (0 = CSV only, no volumes)')
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    evaluate(
        cfg_file,
        checkpoint      = args.medsam2_ckpt,
        model_cfg       = args.sam2_cfg,
        target_data_dir = args.target_data_dir,
        fold            = args.fold,
        eval_labels     = args.test_label,
        prompt_mode     = args.prompt_mode,
        margin          = args.margin,
        device          = args.device,
        save_dir        = args.save_dir,
        save_topk       = args.save_topk,
        seed            = args.seed,
        refine_iters    = args.refine_iters,
        query_slice     = args.query_slice,
        n_anchors       = args.n_anchors,
        anchor_min_gap  = args.anchor_min_gap,
        support_slices  = args.support_slices,
        support_min_gap = args.support_min_gap,
        single_leg      = args.single_leg,
        cc_mode         = args.cc_mode,
        neg_points      = args.neg_points,
        max_neg_points  = args.max_neg_points,
        split_legs      = args.split_legs,
        anchor_coherence = args.anchor_coherence,
    )
