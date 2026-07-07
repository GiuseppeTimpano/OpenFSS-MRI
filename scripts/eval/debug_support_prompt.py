"""
Debug tool for prompt_mode=support (models/support_prompt.py's dense + body-mask +
similarity-ring path). For every query scan, saves a PNG with the support frame used,
the prompted query frame (GT vs predicted contour, pos/neg points), and per-scan Dice --
to see WHERE the pipeline breaks (bad point placement, bad support-scan pick,
propagation loss) instead of just a Dice number from scores.csv.

Same rng seed + call sequence as evaluate()'s prompt_mode=='support' branch in
eval_medsam2.py, so the support-scan assignments (and Dice) match a matching eval run
one-to-one -- this is a visual companion to that run, not a separate evaluation.
"""
import argparse
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from models.medsam2_adapter import MedSAM2Segmenter, volume_to_uint8
from models.support_prompt import key_slice, support_prompt_for_query_dense_bodymasked_similarity


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_raw(data_dir: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    return img, lbl


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--medsam2_ckpt', required=True)
    ap.add_argument('--sam2_cfg', required=True)
    ap.add_argument('--target_data_dir', required=True)
    ap.add_argument('--test_label', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42,
                    help='must match the eval run to reproduce the same support-scan picks')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    label_names = cfg['data']['label_names']
    label_val = args.test_label
    label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)

    paths = sorted(glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
    query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '') for p in paths]
    query_sids = [s for s in query_sids if not s.startswith('P')]  # exclude pathological
    if not query_sids:
        raise ValueError(f'No query scans found in {args.target_data_dir}')

    os.makedirs(args.out_dir, exist_ok=True)
    seg = MedSAM2Segmenter(args.medsam2_ckpt, args.sam2_cfg, device=args.device)

    rng = random.Random(args.seed + label_val)
    support_fg_idx: dict[str, np.ndarray] = {}
    for sid in query_sids:
        lbl = _read_nii(os.path.join(args.target_data_dir, f'label_{sid}.nii.gz'))
        idx = np.where((lbl == label_val).any(axis=(1, 2)))[0]
        if len(idx):
            support_fg_idx[sid] = idx

    for qsid in query_sids:
        q_img, q_lbl = _load_raw(args.target_data_dir, qsid)
        q_fg = (q_lbl == label_val).astype(np.uint8)
        fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
        if len(fg_idx) == 0:
            print(f'[SKIP] {qsid}: no FG for {label_name}')
            continue

        z0, z1 = int(fg_idx.min()), int(fg_idx.max())
        vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]

        pool = [s for s in support_fg_idx if s != qsid]
        if not pool:
            print(f'[SKIP] {qsid}: no support pool')
            continue
        supp_sid = rng.choice(pool)
        supp_img, supp_lbl = _load_raw(args.target_data_dir, supp_sid)
        supp_fg = (supp_lbl == label_val).astype(np.uint8)
        supp_z = key_slice(supp_fg)
        supp_frame_u8 = volume_to_uint8(supp_img)[supp_z]
        supp_mask2d = supp_fg[supp_z].astype(bool)

        query_frames = [(int(z) - z0, vol_u8[int(z) - z0]) for z in fg_idx]
        frame_idx, pos_xy, neg_xy = support_prompt_for_query_dense_bodymasked_similarity(
            seg, supp_frame_u8, supp_mask2d, query_frames)
        points = {frame_idx: (pos_xy, neg_xy)}
        seg_crop = seg.segment_volume_points(vol_u8, points)

        pred_full = np.zeros_like(q_fg)
        pred_full[z0:z1 + 1] = seg_crop
        d = _dice(pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool))

        prompted_z = z0 + frame_idx
        q_frame_u8 = vol_u8[frame_idx]
        gt2d = q_fg[prompted_z].astype(bool)
        pred2d = seg_crop[frame_idx].astype(bool)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(supp_frame_u8, cmap='gray')
        axes[0].contour(supp_mask2d, colors='yellow', linewidths=1.5)
        axes[0].set_title(f'support: {supp_sid} (z={supp_z})')
        axes[0].axis('off')

        axes[1].imshow(q_frame_u8, cmap='gray')
        axes[1].contour(gt2d, colors='yellow', linewidths=1.5)
        axes[1].contour(pred2d, colors='red', linewidths=1.5)
        axes[1].plot(*pos_xy, 'g*', markersize=16, label='pos')
        axes[1].plot(*neg_xy, 'rx', markersize=14, mew=3, label='neg')
        axes[1].set_title(f'{qsid} prompt-frame z={prompted_z}  Dice={d:.3f}')
        axes[1].legend()
        axes[1].axis('off')

        plt.tight_layout()
        out = os.path.join(args.out_dir, f'{label_name}_{qsid}_dice{d:.3f}.png')
        plt.savefig(out, dpi=120)
        plt.close(fig)
        print(f'{qsid}: Dice={d:.4f}  support={supp_sid}  saved={out}')


if __name__ == '__main__':
    main()
