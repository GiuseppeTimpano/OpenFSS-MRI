"""
Visualization of few-shot segmentation results.

Each row = one query slice.
Columns: Support | Query | Prediction | GT

Usage:
  python viz_seg.py \\
    --config configs/resnet.yaml \\
    --checkpoint lightning_logs/version_0/checkpoints/last.ckpt \\
    --label 1 \\
    --out liver.pdf
"""

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from models.fewshot import FewShotConfig, QNetFewShot, ALPNetFewShot

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

_GT_COLOR   = '#2ECC71'
_PRED_COLOR = '#E74C3C'
_FP_COLOR   = '#E74C3C'
_FN_COLOR   = '#3498DB'
VMIN, VMAX  = -2.5, 2.5
FILL_ALPHA  = 0.50


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
        half = 1.0 / (n_part * 2)
        step = (1.0 - 1.0 / n_part) / (n_part - 1)
        pcts = [half + step * i for i in range(n_part)]
    return (np.array(pcts) * n_fg).astype(int)


def _show(ax, img, mask=None, color=None, title=''):
    ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='bilinear')
    if mask is not None and color is not None:
        rgb  = matplotlib.colors.to_rgb(color)
        rgba = np.zeros((*img.shape, 4), dtype=np.float32)
        rgba[..., :3] = rgb
        rgba[..., 3]  = mask.astype(np.float32) * FILL_ALPHA
        ax.imshow(rgba, interpolation='none')
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def run_inference(model, device, sup_imgs_list, sup_masks_list, q_fg_img, n_part):
    C_q = len(q_fg_img)
    H, W = q_fg_img.shape[1], q_fg_img.shape[2]
    pred_vol = torch.zeros(C_q, H, W, dtype=torch.long)
    bounds   = np.linspace(0, C_q, n_part + 1).astype(int)

    with torch.no_grad():
        for ci in range(n_part):
            s_img  = sup_imgs_list[ci]
            s_mask = sup_masks_list[ci]
            for j in range(bounds[ci], bounds[ci + 1]):
                qi   = torch.from_numpy(q_fg_img[j]).to(device).unsqueeze(0)
                pred = model(s_img, s_mask, qi)
                pred_vol[j] = pred.argmax(dim=1).cpu().squeeze(0)

    return pred_vol.numpy()


def dice3d(gt, pred):
    tp = ((gt == 1) & (pred == 1)).sum()
    fp = ((gt == 0) & (pred == 1)).sum()
    fn = ((gt == 1) & (pred == 0)).sum()
    return float(2 * tp / (2 * tp + fp + fn + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/resnet.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--label',      type=int, default=1,
                        help='1=LIVER 2=RK 3=LK 4=SPLEEN')
    parser.add_argument('--query_scan', type=str, default=None)
    parser.add_argument('--n_slices',   type=int, default=5,
                        help='number of query rows to show')
    parser.add_argument('--supp_idx',   type=int, default=0)
    parser.add_argument('--n_part',     type=int, default=3)
    parser.add_argument('--out',             type=str, default=None,
                        help='output file (.pdf/.png/.svg); omit for interactive')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='query dataset dir (e.g. AMOS); support always from config data_dir')
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg       = cfg['data']
    model_cfg  = cfg['model']
    model_name = model_cfg['name']
    data_dir   = dcfg['data_dir']
    label_name = dcfg['label_names'][args.label]

    device = torch.device(args.device)
    fcfg = FewShotConfig(
        encoder_type = model_name,
        n_shot       = dcfg['n_shot'],
        backbone     = model_cfg.get('backbone', 'resnet'),
        arch         = model_cfg.get('arch', 'vit'),
        model_name   = model_cfg.get('model_name', 'dinov3_vitb16'),
        weights_path = model_cfg.get('weights_path'),
        repo_dir     = model_cfg.get('repo_dir'),
        lora_rank    = model_cfg.get('lora_rank', 0),
    )
    bw     = cfg.get('train', {}).get('bg_loss_weight', 0.1)
    model  = QNetFewShot(fcfg, bg_loss_weight=bw) if model_name == 'qnet' \
             else ALPNetFewShot(fcfg, bg_loss_weight=bw)

    raw = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in raw:
        state = {k.removeprefix('_model.'): v
                 for k, v in raw['state_dict'].items() if k.startswith('_model.')}
    else:
        state = raw
    model.load_state_dict(state)
    model.to(device).eval()

    _, test_ids = get_fold_ids(data_dir, dcfg['fold'], dcfg['n_folds'])
    supp_sid    = test_ids[args.supp_idx]

    query_data_dir = args.target_data_dir or data_dir
    if args.target_data_dir:
        import glob as _glob
        _paths     = sorted(_glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
        query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '') for p in _paths]
    else:
        query_sids = [s for s in test_ids if s != supp_sid]

    qsid = args.query_scan or query_sids[0]
    print(f'support={supp_sid}  query={qsid}  organ={label_name}')

    supp_img, supp_lbl = _load_scan(data_dir, supp_sid)
    q_img,    q_lbl    = _load_scan(query_data_dir, qsid)

    supp_fg = (supp_lbl == args.label).astype(np.float32)
    q_fg    = (q_lbl    == args.label).astype(np.float32)

    supp_fg_idx = np.where(supp_fg.any(axis=(1, 2)))[0]
    q_fg_idx    = np.where(q_fg.any(axis=(1, 2)))[0]

    sel_z     = supp_fg_idx[_support_indices(args.n_part, len(supp_fg_idx))]
    sup_imgs  = [torch.from_numpy(supp_img[z]).to(device).unsqueeze(0).unsqueeze(0) for z in sel_z]
    sup_masks = [torch.from_numpy(supp_fg[z]).to(device).unsqueeze(0).unsqueeze(0)  for z in sel_z]

    q_fg_img = q_img[q_fg_idx]
    pred_np  = run_inference(model, device, sup_imgs, sup_masks, q_fg_img, args.n_part)
    q_gt_np  = q_fg[q_fg_idx]

    dsc = dice3d(q_gt_np, pred_np)
    print(f'3D Dice: {dsc:.4f}')

    # evenly spaced query slices to display
    display_idx = np.linspace(0, len(q_fg_idx) - 1, args.n_slices, dtype=int)

    # support reference slice (middle of sel_z)
    supp_ref_z = sel_z[args.n_part // 2]

    N_ROWS = args.n_slices
    N_COLS = 4
    COL_W  = 2.2
    ROW_H  = 2.2

    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * COL_W, N_ROWS * ROW_H),
                             facecolor='white',
                             constrained_layout=True)

    # ensure 2D even for n_slices=1
    if N_ROWS == 1:
        axes = axes[np.newaxis, :]

    for row_i, qi in enumerate(display_idx):
        z_abs   = q_fg_idx[qi]
        q_sl    = q_fg_img[qi]
        gt_sl   = q_gt_np[qi]
        pred_sl = pred_np[qi]

        # support slice: pick the chunk that covers this query index
        chunk   = int(qi / max(len(q_fg_idx) - 1, 1) * (args.n_part - 1) + 0.5)
        chunk   = min(chunk, args.n_part - 1)
        s_z     = sel_z[chunk]

        # column headers only on row 0; z-labels as ylabel on every row
        _show(axes[row_i, 0], supp_img[s_z], supp_fg[s_z], _GT_COLOR,
              title='Support' if row_i == 0 else '')
        _show(axes[row_i, 1], q_sl,
              title='Query' if row_i == 0 else '')
        _show(axes[row_i, 2], q_sl, pred_sl, _PRED_COLOR,
              title='Prediction' if row_i == 0 else '')
        _show(axes[row_i, 3], q_sl, gt_sl, _GT_COLOR,
              title='Ground Truth' if row_i == 0 else '')

        # row label: support z on col 0, query z on col 1
        axes[row_i, 0].set_ylabel(f's z={s_z}', fontsize=7,
                                  rotation=0, labelpad=28, va='center')
        axes[row_i, 1].set_ylabel(f'q z={z_abs}', fontsize=7,
                                  rotation=0, labelpad=28, va='center')

    fig.suptitle(
        f'{model_name.upper()} · {label_name} · '
        f'supp: {supp_sid} · query: {qsid} · Dice = {dsc:.3f}',
        fontsize=10, fontweight='bold',
    )

    legend_patches = [
        mpatches.Patch(facecolor=_GT_COLOR,   alpha=0.7, label='GT / Support mask'),
        mpatches.Patch(facecolor=_PRED_COLOR, alpha=0.7, label='Prediction'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2,
               frameon=False, fontsize=8)

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        fmt = 'pdf' if ext == '.pdf' else ('svg' if ext == '.svg' else 'png')
        plt.savefig(args.out, format=fmt, facecolor='white')
        print(f'saved → {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
