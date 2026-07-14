"""Muscle_Basel_Data preprocessing (proprietary calf MRI, 4 scans).

Input:  data/datasets/Muscle_Basel_Data/data{1..4}.npz
        keys: data [H,W,Z] float32, resolution [x,y,z] mm, comment (scan id),
        mask_<Muscle Name>_<L|R> [H,W,Z] uint8 (one array per muscle/side).
Output: data/datasets/Muscle_Basel_Data/processed/{image_<scan>.nii.gz, label_<scan>.nii.gz}

No N4/z-score here (unlike data/datasets/mri_muscle.py): eval_medsam2.py's
volume_to_uint8 windows on percentiles, which is invariant to the z-score's
affine rescale, and this data has no visible bias-field artifact to justify N4.
Only an upper-percentile clip (matches offline_preprocessing.intensity_clip_upper)
to tame outlier voxels before MedSAM2's own [0,255] windowing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

MUSCLE_BASEL_LABEL_NAMES = [
    "BG",
    "L_SOL", "L_GM", "L_GL", "L_TA", "L_ELD", "L_PER",
    "R_SOL", "R_GM", "R_GL", "R_TA", "R_ELD", "R_PER",
]

# npz mask-key suffix -> label id (order fixed, matches MUSCLE_BASEL_LABEL_NAMES)
_MUSCLE_CODE = {
    "Soleus": "SOL",
    "Gastrocnemius Medialis": "GM",
    "Gastrocnemius Lateralis": "GL",
    "Tibialis Anterior": "TA",
    "Extensor Longus Digitorum": "ELD",
    "Peroneus": "PER",
}
MUSCLE_BASEL_LABEL_MAP = {
    f"{side}_{code}": MUSCLE_BASEL_LABEL_NAMES.index(f"{side}_{code}")
    for muscle, code in _MUSCLE_CODE.items()
    for side in ("L", "R")
}


def _to_sitk(arr_hwz: np.ndarray, spacing_xyz: np.ndarray) -> sitk.Image:
    """[H,W,Z] -> sitk.Image [Z,Y,X] with (x,y,z) spacing."""
    arr_zhw = np.moveaxis(arr_hwz, -1, 0)
    img = sitk.GetImageFromArray(arr_zhw)
    img.SetSpacing(tuple(float(v) for v in spacing_xyz))
    return img


def load_npz_scan(npz_path: Path) -> tuple[sitk.Image, sitk.Image, str]:
    """One Muscle_Basel data{i}.npz -> (image, merged multi-label mask, scan_id)."""
    d = np.load(npz_path, allow_pickle=True)
    scan_id = str(d["comment"])
    spacing = d["resolution"]

    img = _to_sitk(d["data"].astype(np.float32), spacing)

    out = np.zeros(np.moveaxis(d["data"], -1, 0).shape, dtype=np.uint8)
    for key in d.files:
        if not key.startswith("mask_"):
            continue
        code = key[len("mask_"):]
        muscle, side = code.rsplit("_", 1)
        label = MUSCLE_BASEL_LABEL_MAP.get(f"{side}_{_MUSCLE_CODE[muscle]}")
        if label is None:
            continue
        m = np.moveaxis(d[key], -1, 0)
        out[m > 0] = label

    lbl = sitk.GetImageFromArray(out)
    lbl.CopyInformation(img)
    return img, lbl, scan_id


def preprocess_muscle_basel(
    raw_dir: Path,
    out_dir: Path,
    upper_percentile: float = 99.5,
) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in sorted(raw_dir.glob("data*.npz")):
        img, lbl, scan_id = load_npz_scan(npz_path)

        arr = sitk.GetArrayFromImage(img)
        high = np.percentile(arr, upper_percentile)
        arr = np.clip(arr, None, high).astype(np.float32)
        img_clipped = sitk.GetImageFromArray(arr)
        img_clipped.CopyInformation(img)

        sitk.WriteImage(img_clipped, str(out_dir / f"image_{scan_id}.nii.gz"))
        sitk.WriteImage(lbl, str(out_dir / f"label_{scan_id}.nii.gz"))
        print(f"  {scan_id} -> {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Muscle_Basel_Data preprocessing")
    parser.add_argument("--raw_dir", type=Path,
                        default=Path("data/datasets/Muscle_Basel_Data"))
    parser.add_argument("--out_dir", type=Path,
                        default=Path("data/datasets/Muscle_Basel_Data/processed"))
    args = parser.parse_args()

    preprocess_muscle_basel(args.raw_dir, args.out_dir)
