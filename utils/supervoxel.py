"""
Supervoxel generation for MRI/CT volumes (preprocessing step).

Faithful port of the original Q-Net / SSL-ALPNet pipeline
(data/supervoxels/generate_supervoxels.py from the Q-Net repo):

  * 3D Felzenszwalb graph-based segmentation over the whole volume
    (NOT slice-by-slice) — so a supervoxel id is a coherent 3D blob,
    consistent across adjacent slices. This is what makes the
    "neighbours" episode sampling (support/query = adjacent slices of the
    same scan) a well-posed self-supervised task.
  * intensity rescaled to 0..255, sigma=0, min_size=n_sv, anisotropic
    edge weighting from voxel spacing.
  * per-slice foreground body mask, then background supervoxels zeroed.

Build the 3D kernel once before running this (see utils/felzenszwalb_3d):
    cd utils/felzenszwalb_3d && pip install cython && python setup.py build_ext --inplace

Usage:
    python -m utils.supervoxel --data_dir /path/to/data --preset CHAOST2
    python -m utils.supervoxel --data_dir /path/to/data --n_sv 5000 --fg_thresh 10

Expected input:
    data_dir/
        image_<id>.nii.gz

Output (written to out_dir, defaults to data_dir):
    out_dir/
        superpix-<label_prefix>_<id>.nii.gz
        fgmask_<id>.nii.gz
        classmap_0.json     (any foreground pixel counts)
        classmap_1.json     (label covers >= 1% of slice pixels)
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_fill_holes
from skimage.measure import label as cc_label

from utils.felzenszwalb_3d import felzenszwalb_3d


# ──────────────────────────────────────────────────────────────────────────────
# Optional presets for known datasets.
# fg_thresh: intensity (on the 0..255 rescaled image) below which a pixel is
#            treated as air/background. The original uses 10 for CHAOST2/CMR.
# ──────────────────────────────────────────────────────────────────────────────
PRESETS: dict[str, dict] = {
    'CHAOST2': {'fg_thresh': 10.0},
    'CMR':     {'fg_thresh': 10.0},
    'SABS':    {'fg_thresh': 1e-4},
}


@dataclass
class SupervoxelConfig:
    n_sv:         int   = 5000      # Felzenszwalb min_size (original: n_sv=5000)
    sigma:        float = 0.0       # Gaussian smoothing before Felzenszwalb (original: 0)
    fg_thresh:    float = 10.0      # body mask intensity threshold on 0..255 image
    label_prefix: str   = 'MIDDLE'  # output filename: superpix-<prefix>_<id>.nii.gz


# ──────────────────────────────────────────────────────────────────────────────
# Core functions (ported from generate_supervoxels.py)
# ──────────────────────────────────────────────────────────────────────────────

def fg_mask2d(img_2d: np.ndarray, thresh: float) -> np.ndarray:
    """
    Binary foreground (body) mask for one axial slice.

    1. Threshold: pixels above thresh = foreground candidate.
    2. Keep only the largest connected component — removes scattered noise.
    3. Fill internal holes — dark organs inside the body boundary are kept.
    """
    mask = np.float32(img_2d > thresh)

    if mask.max() < 0.999:
        return mask  # no foreground on this slice (e.g. top/bottom of volume)

    labeled = cc_label(mask)
    largest = labeled == (np.argmax(np.bincount(labeled.flat)[1:]) + 1)
    filled  = binary_fill_holes(largest)
    return np.float32(filled)


def supervox_masking(seg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Remove supervoxels in the background region (ported verbatim).

    Shift Felzenszwalb's 0-label up so it is not confused with the enforced
    background label (also 0), then zero out everything outside the body mask.
    Supervoxel ids stay consistent across slices (3D-coherent).
    """
    seg = np.int32(seg)
    seg[seg == 0] = seg.max() + 1
    seg[mask == 0] = 0
    return seg


def supervoxel_volume(
    img255: np.ndarray,
    spacing_zxy: tuple,
    cfg: SupervoxelConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate 3D-coherent supervoxel labels for one volume.

    Args:
        img255:      float32 [D, H, W], intensity rescaled to 0..255
        spacing_zxy: (z, x, y) voxel spacing for anisotropic edge weighting
        cfg:         SupervoxelConfig

    Returns:
        fg_mask_vol: [D, H, W] float32, per-slice body mask
        seg_vol:     [D, H, W] int32,   supervoxel labels (0 = background)
    """
    seg = felzenszwalb_3d(img255, min_size=cfg.n_sv, sigma=cfg.sigma, spacing=spacing_zxy)

    fg_mask_vol = np.zeros(seg.shape, dtype=np.float32)
    for z in range(seg.shape[0]):
        fg_mask_vol[z] = fg_mask2d(img255[z], cfg.fg_thresh)

    seg_vol = supervox_masking(seg, fg_mask_vol)
    return fg_mask_vol, seg_vol


def build_classmap(
    scan_labels:   dict[str, np.ndarray],
    min_fg_ratio:  float = 0.0,
) -> dict:
    """
    Build a lookup index: supervoxel label → {scan_id → [slice indices]}.

    Ids are per-volume (Felzenszwalb relabels each volume independently), so the
    same integer in two scans is NOT the same region. Training therefore samples
    support and query from the SAME scan; this index just lists, per scan, which
    slices contain each supervoxel.

    Returns dict shaped as:
        { "5": {"1": [5, 6, 7], "2": [22]}, "8": {...}, ... }
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

    # rescale to 0..255 (matches original generate_supervoxels.py)
    img255 = 255.0 * (img - img.min()) / (np.ptp(img) + 1e-8)

    # sitk spacing is (x, y, z); the kernel wants (z, x, y)
    sx, sy, sz = im_obj.GetSpacing()
    spacing_zxy = (sz, sx, sy)

    fg_mask_vol, seg_vol = supervoxel_volume(img255, spacing_zxy, cfg)

    sitk.WriteImage(
        _copy_sitk_meta(im_obj, seg_vol),
        os.path.join(out_dir, f'superpix-{cfg.label_prefix}_{scan_id}.nii.gz'),
    )
    sitk.WriteImage(
        _copy_sitk_meta(im_obj, fg_mask_vol),
        os.path.join(out_dir, f'fgmask_{scan_id}.nii.gz'),
    )

    n_sv = len(np.unique(seg_vol[seg_vol > 0]))
    print(f'  scan {scan_id}: {n_sv} supervoxels '
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

    print(f'{len(img_paths)} scans | n_sv={cfg.n_sv} sigma={cfg.sigma} '
          f'fg_thresh={cfg.fg_thresh} prefix={cfg.label_prefix}')

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
        description='Generate 3D Felzenszwalb supervoxel pseudo-labels for MRI/CT volumes.'
    )
    parser.add_argument('--data_dir',     required=True,
                        help='Directory containing image_*.nii.gz files')
    parser.add_argument('--out_dir',      default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--preset',       default=None, choices=list(PRESETS.keys()),
                        help='Dataset preset — sets fg_thresh automatically')
    parser.add_argument('--n_sv',         type=int,   default=5000,
                        help='Felzenszwalb min_size / target supervoxel size (default: 5000)')
    parser.add_argument('--sigma',        type=float, default=0.0,
                        help='Gaussian smoothing sigma (default: 0)')
    parser.add_argument('--fg_thresh',    type=float, default=None,
                        help='Body-mask intensity threshold on 0..255 image (overrides preset)')
    parser.add_argument('--label_prefix', type=str,   default='MIDDLE',
                        help='Output filename prefix (default: MIDDLE)')
    args = parser.parse_args()

    # resolve fg_thresh: explicit arg > preset > default 10
    if args.fg_thresh is not None:
        fg_thresh = args.fg_thresh
    elif args.preset is not None:
        fg_thresh = PRESETS[args.preset]['fg_thresh']
    else:
        fg_thresh = 10.0

    cfg = SupervoxelConfig(
        n_sv         = args.n_sv,
        sigma        = args.sigma,
        fg_thresh    = fg_thresh,
        label_prefix = args.label_prefix,
    )
    run(
        data_dir = args.data_dir,
        out_dir  = args.out_dir or args.data_dir,
        cfg      = cfg,
    )
