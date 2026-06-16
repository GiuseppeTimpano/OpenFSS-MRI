"""
Visualize few-shot segmentation: support slice, query slice, GT, prediction.

Usage:
  python viz_seg.py \
    --config configs/default.yaml \
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \
    --label 1 \          # organ: 1=LIVER 2=RK 3=LK 4=SPLEEN
    --query_scan CT0001 \
    --out seg_viz.png    # omit to show interactively
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from models.fewshot import FewShotConfig, QNetFewShot, ALPNetFewShot


def _read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)


def _load_scan(data_dir, sid):
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz'))
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img, lbl


def _support_indices(n_part, n_fg):
    if n_part == 1:
        pcts = [0.5]
    else:
        half_part     = 1.0 / (n_part * 2)
        part_interval = (1.0 - 1.0 / n_part) / (n_part - 1)
        pcts = [half_part + part_interval * i for i in range(n_part)]
    return (np.array(pcts) * n_fg).astype(int)


def _overlay(ax, img_slice, mask, color, alpha=0.45, title=''):
    """Show grayscale image with colored mask overlay."""
    ax.imshow(img_slice, cmap='gray', vmin=-2, vmax=2)
    rgba = np.zeros((*img_slice.shape, 4), dtype=np.float32)
    rgba[..., :3] = matplotlib.colors.to_rgb(color)
    rgba[..., 3]  = mask.astype(np.float32) * alpha
    ax.imshow(rgba)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',      type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--label',       type=int, default=1,
                        help='organ label index (1=LIVER 2=RK 3=LK 4=SPLEEN)')
    parser.add_argument('--query_scan',  type=str, default=None,
                        help='scan ID to use as query (default: first non-support test scan)')
    parser.add_argument('--n_slices',    type=int, default=5,
                        help='number of query slices to display (evenly spaced in FG)')
    parser.add_argument('--supp_idx',    type=int, default=0)
    parser.add_argument('--n_part',      type=int, default=3)
    parser.add_argument('--out',         type=str, default=None,
                        help='save path (e.g. viz.png); if omitted, show window')
    parser.add_argument('--device',      type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    data_cfg   = cfg_file['data']
    model_name = cfg_file['model']['name']
    data_dir   = data_cfg['data_dir']
    fold       = data_cfg['fold']
    n_folds    = data_cfg['n_folds']
    n_shot     = data_cfg['n_shot']
    label_names = data_cfg['label_names']
    label_name  = label_names[args.label] if args.label < len(label_names) else str(args.label)

    # build + load model
    device = torch.device(args.device)
    cfg    = FewShotConfig(encoder_type=model_name, n_shot=n_shot)
    bg_loss_weight = cfg_file.get('train', {}).get('bg_loss_weight', 0.1)
    if model_name == 'qnet':
        model = QNetFewShot(cfg, bg_loss_weight=bg_loss_weight)
    else:
        model = ALPNetFewShot(cfg, bg_loss_weight=bg_loss_weight)

    raw = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in raw:
        state = {k.removeprefix('_model.'): v
                 for k, v in raw['state_dict'].items()
                 if k.startswith('_model.')}
    else:
        state = raw
    model.load_state_dict(state)
    model.to(device).eval()

    # pick scans
    _, test_ids = get_fold_ids(data_dir, fold, n_folds)
    supp_sid   = test_ids[args.supp_idx]
    query_sids = [s for s in test_ids if s != supp_sid]
    qsid       = args.query_scan if args.query_scan else query_sids[0]

    print(f'support={supp_sid}  query={qsid}  label={label_name}({args.label})')

    supp_img, supp_lbl = _load_scan(data_dir, supp_sid)
    q_img,    q_lbl    = _load_scan(data_dir, qsid)

    supp_fg_mask = (supp_lbl == args.label).astype(np.float32)
    q_fg_mask    = (q_lbl    == args.label).astype(np.float32)

    supp_fg_idx = np.where(supp_fg_mask.any(axis=(1, 2)))[0]
    q_fg_idx    = np.where(q_fg_mask.any(axis=(1, 2)))[0]

    # support slices (n_part evenly spaced)
    sel_idx = _support_indices(args.n_part, len(supp_fg_idx))
    sel_z   = supp_fg_idx[sel_idx]

    sup_imgs_list  = [torch.from_numpy(supp_img[z]).to(device).unsqueeze(0).unsqueeze(0) for z in sel_z]
    sup_masks_list = [torch.from_numpy(supp_fg_mask[z]).to(device).unsqueeze(0).unsqueeze(0) for z in sel_z]

    # run inference on all FG query slices
    q_fg_img  = q_img[q_fg_idx]
    C_q       = len(q_fg_idx)
    H, W      = q_fg_img.shape[1], q_fg_img.shape[2]
    pred_vol  = torch.zeros(C_q, H, W, dtype=torch.long)
    chunk_bounds = np.linspace(0, C_q, args.n_part + 1).astype(int)

    with torch.no_grad():
        for chunk_i in range(args.n_part):
            s_img  = sup_imgs_list[chunk_i]
            s_mask = sup_masks_list[chunk_i]
            a, b   = chunk_bounds[chunk_i], chunk_bounds[chunk_i + 1]
            for j in range(a, b):
                qi   = torch.from_numpy(q_fg_img[j]).to(device).unsqueeze(0)
                pred = model(s_img, s_mask, qi)
                pred_vol[j] = pred.argmax(dim=1).cpu().squeeze(0)

    pred_np = pred_vol.numpy()
    q_gt_np = q_fg_mask[q_fg_idx]

    # 3D Dice for this query scan
    tp = ((q_gt_np == 1) & (pred_np == 1)).sum()
    fp = ((q_gt_np == 0) & (pred_np == 1)).sum()
    fn = ((q_gt_np == 1) & (pred_np == 0)).sum()
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    print(f'3D Dice ({qsid}, {label_name}): {dice:.4f}')

    # pick n_slices evenly spaced in FG for display
    display_idx = np.linspace(0, C_q - 1, args.n_slices, dtype=int)

    # layout: rows = [support_ref | query_image | GT | pred | error]
    # support_ref only shown in column 0 (repeated for visual reference)
    supp_ref_z = sel_z[args.n_part // 2]   # middle support slice

    fig, axes = plt.subplots(5, args.n_slices, figsize=(3 * args.n_slices, 12))
    fig.suptitle(f'{model_name.upper()} | {label_name} | support={supp_sid} → query={qsid} | Dice={dice:.4f}',
                 fontsize=11, y=0.99)

    row_labels = ['Support ref', 'Query', 'GT', 'Pred', 'Error (FP=red FN=blue)']
    for r, rl in enumerate(row_labels):
        axes[r, 0].set_ylabel(rl, fontsize=8, rotation=90, labelpad=4)

    for col, qi in enumerate(display_idx):
        q_slice  = q_fg_img[qi]
        gt_slice = q_gt_np[qi]
        pd_slice = pred_np[qi]

        # row 0: support reference
        _overlay(axes[0, col], supp_img[supp_ref_z], supp_fg_mask[supp_ref_z],
                 color='lime', title=f'supp z={supp_ref_z}')

        # row 1: query image (no mask)
        axes[1, col].imshow(q_slice, cmap='gray', vmin=-2, vmax=2)
        axes[1, col].set_title(f'z={q_fg_idx[qi]}', fontsize=9)
        axes[1, col].axis('off')

        # row 2: GT
        _overlay(axes[2, col], q_slice, gt_slice, color='lime',
                 title=f'GT (z={q_fg_idx[qi]})')

        # row 3: Prediction
        _overlay(axes[3, col], q_slice, pd_slice, color='cyan',
                 title=f'Pred (z={q_fg_idx[qi]})')

        # row 4: error map — FP red, FN blue
        fp_map = ((gt_slice == 0) & (pd_slice == 1)).astype(np.float32)
        fn_map = ((gt_slice == 1) & (pd_slice == 0)).astype(np.float32)
        axes[4, col].imshow(q_slice, cmap='gray', vmin=-2, vmax=2)
        rgba_fp = np.zeros((*q_slice.shape, 4), dtype=np.float32)
        rgba_fp[..., 0] = 1.0; rgba_fp[..., 3] = fp_map * 0.6
        rgba_fn = np.zeros((*q_slice.shape, 4), dtype=np.float32)
        rgba_fn[..., 2] = 1.0; rgba_fn[..., 3] = fn_map * 0.6
        axes[4, col].imshow(rgba_fp)
        axes[4, col].imshow(rgba_fn)
        axes[4, col].set_title(f'FP={fp_map.sum():.0f} FN={fn_map.sum():.0f}', fontsize=8)
        axes[4, col].axis('off')

    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches='tight')
        print(f'saved → {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
