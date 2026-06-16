"""
Publication-quality visualization of few-shot segmentation results.

Layout (per query scan):
  Row 0      : support reference slice (full width)
  Rows 1..N  : n_slices query slices, 4 columns each
               [Image | GT | Prediction | Error (FP/FN)]

Usage:
  python viz_seg.py \\
    --config configs/default.yaml \\
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

# ── publication style ────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         8,
    'axes.titlesize':    8,
    'axes.labelsize':    8,
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.02,
})

# colour palette (colorblind-safe)
_GT_COLOR   = '#2ECC71'   # emerald green
_PRED_COLOR = '#E74C3C'   # alizarin red
_TP_COLOR   = '#F39C12'   # orange  (used in combined map)
_FP_COLOR   = '#E74C3C'
_FN_COLOR   = '#3498DB'   # peter river blue

VMIN, VMAX = -2.5, 2.5    # z-score window for CT display


# ── helpers ──────────────────────────────────────────────────────────────────

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
        half  = 1.0 / (n_part * 2)
        step  = (1.0 - 1.0 / n_part) / (n_part - 1)
        pcts  = [half + step * i for i in range(n_part)]
    return (np.array(pcts) * n_fg).astype(int)


def _show_img(ax, img, title='', zlabel=None):
    ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='bilinear')
    ax.set_title(title, pad=3)
    if zlabel is not None:
        ax.set_ylabel(zlabel, fontsize=7, labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _overlay_mask(ax, img, mask, hex_color, fill_alpha=0.25, lw=1.2, title='', zlabel=None):
    """Image + filled mask + boundary contour."""
    ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='bilinear')

    # filled tint
    rgb  = matplotlib.colors.to_rgb(hex_color)
    rgba = np.zeros((*img.shape, 4), dtype=np.float32)
    rgba[..., :3] = rgb
    rgba[..., 3]  = mask.astype(np.float32) * fill_alpha
    ax.imshow(rgba, interpolation='none')

    # boundary contour (only if mask has any foreground)
    if mask.any():
        ax.contour(mask, levels=[0.5], colors=[hex_color],
                   linewidths=[lw], alpha=0.95)

    ax.set_title(title, pad=3)
    if zlabel is not None:
        ax.set_ylabel(zlabel, fontsize=7, labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _error_map(ax, img, gt, pred, title='', zlabel=None):
    """FP (red) + FN (blue) overlay on grayscale image."""
    ax.imshow(img, cmap='gray', vmin=VMIN, vmax=VMAX, interpolation='bilinear')

    fp = ((gt == 0) & (pred == 1)).astype(np.float32)
    fn = ((gt == 1) & (pred == 0)).astype(np.float32)

    def _colored(mask, hex_c, alpha=0.55):
        r = np.zeros((*img.shape, 4), dtype=np.float32)
        r[..., :3] = matplotlib.colors.to_rgb(hex_c)
        r[..., 3]  = mask * alpha
        return r

    ax.imshow(_colored(fp, _FP_COLOR), interpolation='none')
    ax.imshow(_colored(fn, _FN_COLOR), interpolation='none')

    n_fp, n_fn = int(fp.sum()), int(fn.sum())
    ax.set_title(f'FP {n_fp} · FN {n_fn}', pad=3)
    if zlabel is not None:
        ax.set_ylabel(zlabel, fontsize=7, labelpad=3)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ── inference ─────────────────────────────────────────────────────────────────

def run_inference(model, device, sup_imgs_list, sup_masks_list,
                  q_fg_img, n_part):
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--label',      type=int, default=1,
                        help='organ label index (1=LIVER 2=RK 3=LK 4=SPLEEN)')
    parser.add_argument('--query_scan', type=str, default=None)
    parser.add_argument('--n_slices',   type=int, default=5,
                        help='query rows to display (evenly spaced in FG)')
    parser.add_argument('--supp_idx',   type=int, default=0)
    parser.add_argument('--n_part',     type=int, default=3)
    parser.add_argument('--out',        type=str, default=None,
                        help='output file (pdf/png/svg); omit for interactive')
    parser.add_argument('--device',     type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── config ──
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg        = cfg['data']
    model_name  = cfg['model']['name']
    data_dir    = dcfg['data_dir']
    label_names = dcfg['label_names']
    label_name  = label_names[args.label] if args.label < len(label_names) else str(args.label)

    # ── model ──
    device = torch.device(args.device)
    fcfg   = FewShotConfig(encoder_type=model_name, n_shot=dcfg['n_shot'])
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

    # ── scans ──
    _, test_ids = get_fold_ids(data_dir, dcfg['fold'], dcfg['n_folds'])
    supp_sid    = test_ids[args.supp_idx]
    query_sids  = [s for s in test_ids if s != supp_sid]
    qsid        = args.query_scan or query_sids[0]
    print(f'support={supp_sid}  query={qsid}  organ={label_name}')

    supp_img, supp_lbl = _load_scan(data_dir, supp_sid)
    q_img,    q_lbl    = _load_scan(data_dir, qsid)

    supp_fg = (supp_lbl == args.label).astype(np.float32)
    q_fg    = (q_lbl    == args.label).astype(np.float32)

    supp_fg_idx = np.where(supp_fg.any(axis=(1, 2)))[0]
    q_fg_idx    = np.where(q_fg.any(axis=(1, 2)))[0]

    sel_z = supp_fg_idx[_support_indices(args.n_part, len(supp_fg_idx))]
    sup_imgs  = [torch.from_numpy(supp_img[z]).to(device).unsqueeze(0).unsqueeze(0) for z in sel_z]
    sup_masks = [torch.from_numpy(supp_fg[z]).to(device).unsqueeze(0).unsqueeze(0)  for z in sel_z]

    # ── inference ──
    q_fg_img = q_img[q_fg_idx]
    pred_np  = run_inference(model, device, sup_imgs, sup_masks, q_fg_img, args.n_part)
    q_gt_np  = q_fg[q_fg_idx]

    dsc = dice3d(q_gt_np, pred_np)
    print(f'3D Dice: {dsc:.4f}')

    # ── display slice selection ──
    display_idx = np.linspace(0, len(q_fg_idx) - 1, args.n_slices, dtype=int)

    # ── figure layout ──────────────────────────────────────────────────────
    # 1 header row (support) + n_slices query rows
    # 4 content columns:  Image | GT | Prediction | Error
    N_ROWS = 1 + args.n_slices
    N_COLS = 4
    COL_W  = 1.6   # inches per column
    ROW_H  = 1.6   # inches per row
    HEADER_RATIO = 0.85   # support row slightly shorter

    fig = plt.figure(figsize=(N_COLS * COL_W, ROW_H * (N_ROWS - 1 + HEADER_RATIO)),
                     facecolor='white')

    height_ratios = [HEADER_RATIO] + [1.0] * args.n_slices
    gs = fig.add_gridspec(N_ROWS, N_COLS,
                          hspace=0.06, wspace=0.03,
                          height_ratios=height_ratios)

    # ── column headers ──
    col_titles = ['Image', f'GT ({label_name})', 'Prediction', 'Error map']
    for c, ct in enumerate(col_titles):
        ax_h = fig.add_subplot(gs[0, c])
        # top support row: col 0 = support reference, cols 1-3 = spacer/label
        if c == 0:
            supp_ref_z = sel_z[args.n_part // 2]
            _overlay_mask(ax_h, supp_img[supp_ref_z], supp_fg[supp_ref_z],
                          _GT_COLOR, fill_alpha=0.25, lw=1.2,
                          title=f'Support · z={supp_ref_z}')
        else:
            ax_h.set_visible(False)

    # column title text (above first data row)
    for c, ct in enumerate(col_titles):
        fig.text(
            (c + 0.5) / N_COLS, 1.0,
            ct,
            ha='center', va='bottom',
            fontsize=8, fontweight='bold',
            transform=fig.transFigure
        )

    # ── query rows ──
    for row_i, qi in enumerate(display_idx):
        z_abs    = q_fg_idx[qi]
        q_sl     = q_fg_img[qi]
        gt_sl    = q_gt_np[qi]
        pred_sl  = pred_np[qi]

        z_label = f'z = {z_abs}'

        ax_img  = fig.add_subplot(gs[row_i + 1, 0])
        ax_gt   = fig.add_subplot(gs[row_i + 1, 1])
        ax_pred = fig.add_subplot(gs[row_i + 1, 2])
        ax_err  = fig.add_subplot(gs[row_i + 1, 3])

        _show_img    (ax_img,  q_sl,                title='', zlabel=z_label)
        _overlay_mask(ax_gt,   q_sl, gt_sl,   _GT_COLOR,   fill_alpha=0.25, lw=1.2)
        _overlay_mask(ax_pred, q_sl, pred_sl, _PRED_COLOR, fill_alpha=0.25, lw=1.2)
        _error_map   (ax_err,  q_sl, gt_sl, pred_sl)

    # ── legend & title ──
    legend_patches = [
        mpatches.Patch(facecolor=_GT_COLOR,   edgecolor='none', alpha=0.7, label='GT'),
        mpatches.Patch(facecolor=_PRED_COLOR, edgecolor='none', alpha=0.7, label='Prediction'),
        mpatches.Patch(facecolor=_FP_COLOR,   edgecolor='none', alpha=0.7, label='False positive'),
        mpatches.Patch(facecolor=_FN_COLOR,   edgecolor='none', alpha=0.7, label='False negative'),
    ]
    fig.legend(handles=legend_patches,
               loc='lower center', ncol=4,
               frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f'{model_name.upper()} · {label_name} · '
        f'support: {supp_sid} → query: {qsid} · '
        f'3D Dice = {dsc:.3f}',
        fontsize=9, fontweight='bold', y=1.02
    )

    if args.out:
        ext = os.path.splitext(args.out)[1].lower()
        fmt = 'pdf' if ext == '.pdf' else ('svg' if ext == '.svg' else 'png')
        plt.savefig(args.out, format=fmt, facecolor='white')
        print(f'saved → {args.out}')
    else:
        plt.show()


if __name__ == '__main__':
    main()
