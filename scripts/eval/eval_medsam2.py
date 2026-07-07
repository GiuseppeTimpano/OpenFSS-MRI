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
bbox, see models/support_prompt.py:support_prompt_for_query_dense_bodymasked_bbox).

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
from models.support_prompt import key_slice, support_prompt_for_query_dense_bodymasked_bbox


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


def evaluate(cfg: dict, checkpoint: str, model_cfg: str,
             target_data_dir: str | None, fold: int | None,
             eval_labels: list[int] | None, prompt_mode: str, margin: int,
             device: str, save_dir: str | None, save_topk: int = 1,
             seed: int = 42, refine_iters: int = 0) -> dict:
    data_cfg    = cfg['data']
    data_dir    = data_cfg['data_dir']
    n_folds     = data_cfg['n_folds']
    label_names = data_cfg['label_names']
    if eval_labels is None:
        eval_labels = list(range(1, len(label_names)))

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
        if prompt_mode == 'support_bbox':
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
                supp_z = key_slice(supp_fg)
                supp_frame_u8 = volume_to_uint8(supp_img)[supp_z]
                supp_mask2d = supp_fg[supp_z].astype(bool)

                query_frames = [(int(z) - z0, vol_u8[int(z) - z0]) for z in fg_idx]
                frame_idx, box = support_prompt_for_query_dense_bodymasked_bbox(
                    seg, supp_frame_u8, supp_mask2d, query_frames)
                seg_crop = seg.segment_volume(vol_u8, {frame_idx: np.asarray(box, dtype=np.float32)},
                                               refine_iters=refine_iters)
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
                        choices=['perslice', 'key', 'support_bbox'])
    parser.add_argument('--margin',          type=int, default=3,
                        help='oracle box margin in pixels')
    parser.add_argument('--seed',            type=int, default=42,
                        help='random support-scan selection (prompt_mode=support_bbox)')
    parser.add_argument('--refine_iters',    type=int, default=0,
                        help='cascaded box->mask->box refinement on the seed frame before '
                             'propagation (prompt_mode=support_bbox); 0 = off')
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
    )
