"""
Supervoxel generation for MRI/CT volumes (preprocessing step).

Ports the approach from ALPNet (data/pseudolabel_gen.ipynb):
  Felzenszwalb graph-based segmentation applied slice-by-slice (2D).

Usage:
    # with a known dataset preset (sets fg_thresh automatically)
    python -m utils.supervoxel --data_dir /path/to/data --preset CHAOST2

    # fully manual (any dataset)
    python -m utils.supervoxel --data_dir /path/to/data --fg_thresh 50 --min_size 400 --sigma 1.0

    # custom output directory and label prefix
    python -m utils.supervoxel --data_dir /path/to/data --out_dir /path/to/out \\
        --label_prefix SMALL --fg_thresh 1e-4 --min_size 200 --sigma 0.8

Expected input:
    data_dir/
        image_<id>.nii.gz    (any integer id)
        ...

Output (written to out_dir, defaults to data_dir):
    out_dir/
        superpix-<label_prefix>_<id>.nii.gz
        fgmask_<id>.nii.gz
        classmap_0.json     (any foreground pixel counts)
        classmap_1.json     (label must cover >= 1% of slice pixels)
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import SimpleITK as sitk
import scipy.ndimage.morphology as snm
from skimage.measure import label as cc_label
from skimage.segmentation import felzenszwalb


# ──────────────────────────────────────────────────────────────────────────────
# Optional presets for known datasets.
# fg_thresh: intensity below this is treated as air/background.
#   MR (CHAOST2): normalized images have near-zero background, but
#                 actual body tissue starts well above zero — use 50.
#   MR (CMR):     similar to CHAOST2 but unnormalized range differs.
#   CT (SABS):    after normalization CT background is very close to 0.
# ──────────────────────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    'CHAOST2': {'fg_thresh': 1e-4 + 50},
    'CMR':     {'fg_thresh': 10.0},
    'SABS':    {'fg_thresh': 1e-4},
}


@dataclass
class SupervoxelConfig:
    fg_thresh:    float = 1e-4      # body mask intensity threshold
    min_size:     int   = 400       # minimum supervoxel size (pixels per slice)
    sigma:        float = 1.0       # Gaussian smoothing before Felzenszwalb
    label_prefix: str   = 'MIDDLE'  # used in output filename: superpix-<prefix>_<id>.nii.gz


# ──────────────────────────────────────────────────────────────────────────────
# Core functions
# ──────────────────────────────────────────────────────────────────────────────

def fg_mask2d(img_2d: np.ndarray, thresh: float) -> np.ndarray:
    """
    Binary foreground mask for one axial slice.

    1. Threshold: pixels above thresh = foreground candidate.
    2. Keep only the largest connected component — removes scattered noise.
    3. Fill internal holes — dark organs (e.g. gallbladder) inside the body
       boundary should be included in the mask even if below threshold.

    Returns float32 binary mask, same spatial shape as img_2d.
    """
    mask = np.float32(img_2d > thresh)

    if mask.max() < 0.999:
        return mask  # no foreground on this slice (e.g. top/bottom of volume)

    labeled = cc_label(mask)
    largest = labeled == (np.argmax(np.bincount(labeled.flat)[1:]) + 1)
    filled  = snm.binary_fill_holes(largest)
    return np.float32(filled)


def _mask_and_reindex(raw_seg2d: np.ndarray, mask2d: np.ndarray) -> np.ndarray:
    """
    Zero out supervoxels that fall in the background, then re-index 1..K.

    Felzenszwalb may assign label 0 to some pixels; we shift it before masking
    to avoid confusing it with the background label we enforce (also 0).

    raw_seg2d: felzenszwalb segmentation
    mask_2d: foreground segmentation
    """
    raw = np.int32(raw_seg2d)
    lbvs   = list(np.unique(raw))
    max_lb = max(lbvs)

    raw[raw == 0] = max_lb + 1   # shift Felzenszwalb's 0 away from bg
    lbvs.append(max_lb)

    raw = raw * np.int32(mask2d)  # pixels outside body → 0

    out   = np.zeros(raw.shape, dtype=np.int32)
    lb_new = 1
    for lbv in lbvs:
        if lbv == 0:
            continue
        out[raw == lbv] = lb_new
        lb_new += 1

    return out


def supervoxel_volume(
    img:       np.ndarray,
    cfg:       SupervoxelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate supervoxel labels for a 3D volume by processing each axial slice.

    Args:
        img:  float32 array [D, H, W], first axis = axial (slice) direction
        cfg:  SupervoxelConfig

    Returns:
        fg_mask_vol:  [D, H, W] float32, binary body mask per slice
        seg_vol:      [D, H, W] int32,   supervoxel labels (0 = background)
    """
    fg_mask_vol = np.zeros(img.shape, dtype=np.float32)
    seg_vol     = np.zeros(img.shape, dtype=np.int32)

    for z in range(img.shape[0]):
        slc     = img[z]
        raw_seg = felzenszwalb(slc, min_size=cfg.min_size, sigma=cfg.sigma)
        fg_mask = fg_mask2d(slc, cfg.fg_thresh)

        seg_vol[z]     = _mask_and_reindex(raw_seg, fg_mask)
        fg_mask_vol[z] = fg_mask

    return fg_mask_vol, seg_vol


def build_classmap(
    scan_labels:   dict[str, np.ndarray],
    min_fg_ratio:  float = 0.0,
) -> dict:
    """
    Build a lookup index: supervoxel label → {scan_id → [slice indices]}.

    Pre-computed at preprocessing time so the dataloader can instantly find
    valid episodes without scanning every volume at runtime.

    Args:
        scan_labels:   {scan_id: seg_vol [D, H, W]}
        min_fg_ratio:  skip (label, slice) pairs where the label covers less
                       than this fraction of the slice pixels.
                       0.0 = any pixel counts.  0.01 = at least 1% coverage.

    Returns dict shaped as:
        { "1": {"003": [5, 6, 7], "007": [22]}, "2": {...}, ... }
    """
    classmap: dict[str, dict[str, list[int]]] = {}

    for scan_id, labels in scan_labels.items():
        n_pixels = labels.shape[1] * labels.shape[2]
        for z in range(labels.shape[0]):
            slc = labels[z]
            for lbl in np.unique(slc[slc > 0]):
                if (slc == lbl).sum() / n_pixels < min_fg_ratio:
                    continue
                key = str(int(lbl))
                classmap.setdefault(key, {}).setdefault(scan_id, []).append(z)

    return classmap


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def _copy_sitk_meta(src: sitk.Image, arr: np.ndarray) -> sitk.Image:
    """Wrap a numpy array in a SimpleITK image, copying spatial metadata from src."""
    out = sitk.GetImageFromArray(arr)
    out.SetSpacing(src.GetSpacing())
    out.SetOrigin(src.GetOrigin())
    out.SetDirection(src.GetDirection())
    return out


def process_scan(
    img_path:  str,
    out_dir:   str,
    cfg:       SupervoxelConfig,
) -> tuple[str, np.ndarray]:
    """Load, segment, and save one NIfTI scan. Returns (scan_id, seg_vol)."""
    scan_id = re.findall(r'\d+', os.path.basename(img_path))[-1]

    im_obj = sitk.ReadImage(img_path)
    img    = sitk.GetArrayFromImage(im_obj).astype(np.float32)  # [D, H, W]

    fg_mask_vol, seg_vol = supervoxel_volume(img, cfg)

    sitk.WriteImage(
        _copy_sitk_meta(im_obj, seg_vol),
        os.path.join(out_dir, f'superpix-{cfg.label_prefix}_{scan_id}.nii.gz'),
    )
    sitk.WriteImage(
        _copy_sitk_meta(im_obj, fg_mask_vol),
        os.path.join(out_dir, f'fgmask_{scan_id}.nii.gz'),
    )

    print(f'  scan {scan_id}: {int(seg_vol.max())} supervoxels '
          f'→ superpix-{cfg.label_prefix}_{scan_id}.nii.gz')
    return scan_id, seg_vol


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run(data_dir: str, out_dir: str, cfg: SupervoxelConfig):
    os.makedirs(out_dir, exist_ok=True)

    img_paths = sorted(
        glob.glob(os.path.join(data_dir, 'image_*.nii.gz')),
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]),
    )
    if not img_paths:
        raise FileNotFoundError(f'No image_*.nii.gz found in {data_dir}')

    print(f'{len(img_paths)} scans | fg_thresh={cfg.fg_thresh} '
          f'min_size={cfg.min_size} sigma={cfg.sigma} prefix={cfg.label_prefix}')

    scan_labels: dict[str, np.ndarray] = {}
    for img_path in img_paths:
        scan_id, seg_vol = process_scan(img_path, out_dir, cfg)
        scan_labels[scan_id] = seg_vol

    for ratio, key in [(0.0, '0'), (0.01, '1')]:
        classmap  = build_classmap(scan_labels, min_fg_ratio=ratio)
        out_path  = os.path.join(out_dir, f'classmap_{key}.json')
        with open(out_path, 'w') as f:
            json.dump(classmap, f)
        print(f'classmap_{key}.json → {len(classmap)} labels')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate Felzenszwalb supervoxel pseudo-labels for MRI/CT volumes.'
    )
    parser.add_argument('--data_dir',     required=True,
                        help='Directory containing image_*.nii.gz files')
    parser.add_argument('--out_dir',      default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--preset',       default=None, choices=list(PRESETS.keys()),
                        help='Dataset preset — sets fg_thresh automatically')
    parser.add_argument('--fg_thresh',    type=float, default=None,
                        help='Intensity threshold for body mask (overrides preset)')
    parser.add_argument('--min_size',     type=int,   default=400,
                        help='Minimum supervoxel size in pixels (default: 400)')
    parser.add_argument('--sigma',        type=float, default=1.0,
                        help='Gaussian smoothing sigma (default: 1.0)')
    parser.add_argument('--label_prefix', type=str,   default='MIDDLE',
                        help='Output filename prefix, e.g. SMALL/MIDDLE/LARGE (default: MIDDLE)')
    args = parser.parse_args()

    # resolve fg_thresh: explicit arg > preset > error
    if args.fg_thresh is not None:
        fg_thresh = args.fg_thresh
    elif args.preset is not None:
        fg_thresh = PRESETS[args.preset]['fg_thresh']
    else:
        parser.error('Provide --preset or --fg_thresh')

    cfg = SupervoxelConfig(
        fg_thresh    = fg_thresh,
        min_size     = args.min_size,
        sigma        = args.sigma,
        label_prefix = args.label_prefix,
    )
    run(
        data_dir = args.data_dir,
        out_dir  = args.out_dir or args.data_dir,
        cfg      = cfg,
    )
