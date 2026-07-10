"""MRI_muscle_2 dataset preprocessing (external figshare thigh/calf dataset).

Input:  data/datasets/MRI_muscle_2/<subject>/Thigh/{Water.nii.gz, mask_muscles.nii.gz}
Output: data/datasets/MRI_muscle_2/processed/WATER/{image_<scan>.nii.gz, label_<scan>.nii.gz}

Single-leg-per-volume dataset (no L/R split, unlike MRI_muscle). Water and
mask_muscles share identical geometry (size/spacing/origin) so no resample is
needed. Only the Thigh region is processed (Calf is present but unused).

Original 13-class scheme is mapped onto 6 muscle types: QF (4 quad heads),
HS (3 hamstring heads), SA (sartorius), GR (gracilis), AD (3 adductor heads
grouped), GLUT (gluteus maximus). Single-leg, no L/R split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.offline_preprocessing import (
    center_crop_2d,
    intensity_clip_upper,
    n4_bias_field_correction,
    z_volume_norm,
)

MRI_MUSCLE_2_LABEL_NAMES = [
    "BG", "QF", "HS", "SA", "GR", "AD", "GLUT",
]

# original figshare label id -> project label id.
# 1-9 map onto the project's QF/HS/SA/GR groups; 10-12 grouped into AD,
# 13 kept as GLUT. No annotation discarded.
RAW_TO_PROJECT_LABEL = {
    1: 1,   # rectus_femoris        -> QF
    2: 1,   # vastus_lateralis      -> QF
    3: 1,   # vastus_intermedius    -> QF
    4: 1,   # vastus_medialis       -> QF
    5: 3,   # sartorius             -> SA
    6: 4,   # gracilis              -> GR
    7: 2,   # biceps_femoris        -> HS
    8: 2,   # semitendinosus        -> HS
    9: 2,   # semimembranosus       -> HS
    10: 5,  # adductor_brevis       -> AD
    11: 5,  # adductor_longus       -> AD
    12: 5,  # adductor_magnus       -> AD
    13: 6,  # gluteus_maximus       -> GLUT
}


def discover_subjects(raw_dir: Path) -> list[str]:
    """Return subject IDs (e.g. '01') that have a Thigh/mask_muscles.nii.gz."""
    raw_dir = Path(raw_dir)
    subjects = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "Thigh" / "mask_muscles.nii.gz").exists() and (path / "Thigh" / "Water.nii.gz").exists():
            subjects.append(path.name)
    return subjects


def remap_labels(mask_img: sitk.Image) -> sitk.Image:
    """Apply RAW_TO_PROJECT_LABEL LUT to a label volume."""
    arr = sitk.GetArrayFromImage(mask_img)
    lut = np.zeros(int(arr.max()) + 1, dtype=np.uint8)
    for raw_id, proj_id in RAW_TO_PROJECT_LABEL.items():
        if raw_id < lut.shape[0]:
            lut[raw_id] = proj_id
    out_arr = lut[arr]
    out = sitk.GetImageFromArray(out_arr)
    out.CopyInformation(mask_img)
    return out


def preprocess_mri_muscle_2(
    raw_dir: Path,
    out_dir: Path,
    echo: str = "Water",
    crop_size: int = 0,
) -> None:
    """Convert raw MRI_muscle_2 Thigh layout into standard image_/label_ NIfTI files."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    echo_out_dir = out_dir / "WATER"
    echo_out_dir.mkdir(parents=True, exist_ok=True)

    for sid in discover_subjects(raw_dir):
        thigh_dir = raw_dir / sid / "Thigh"
        img = sitk.ReadImage(str(thigh_dir / f"{echo}.nii.gz"))
        lbl = sitk.ReadImage(str(thigh_dir / "mask_muscles.nii.gz"))
        lbl = remap_labels(lbl)

        img = n4_bias_field_correction(img)
        img = intensity_clip_upper(img)
        if crop_size > 0:
            padval = float(sitk.GetArrayFromImage(img).min())
            img = center_crop_2d(img, crop_size, padval=padval)
        img = z_volume_norm(img)

        if crop_size > 0:
            lbl = center_crop_2d(lbl, crop_size, padval=0.0)

        sitk.WriteImage(img, str(echo_out_dir / f"image_{sid}.nii.gz"))
        sitk.WriteImage(lbl, str(echo_out_dir / f"label_{sid}.nii.gz"))
        print(f"  {sid} -> {echo_out_dir}")


def build_mri_muscle_2_classmap(
    label_dir: Path,
    label_names: list[str],
    min_pixels_list: list[int],
) -> None:
    """Build gt_classmap JSON files keyed by subject id."""
    import json

    label_dir = Path(label_dir)
    label_paths = sorted(label_dir.glob("label_*.nii.gz"))
    if not label_paths:
        raise FileNotFoundError(f"No label_*.nii.gz in {label_dir}")

    for min_px in min_pixels_list:
        classmap = {name: {} for name in label_names}

        for lbl_path in label_paths:
            scan_id = lbl_path.stem.replace(".nii", "").replace("label_", "")
            lbl_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_path)))

            for name in label_names:
                classmap[name][scan_id] = []

            for z in range(lbl_vol.shape[0]):
                slc = lbl_vol[z]
                slice_sum = int(np.sum(slc))
                if slice_sum < min_px:
                    continue
                present = {int(v) for v in np.unique(slc) if v > 0}
                for cls_idx, name in enumerate(label_names):
                    if cls_idx in present:
                        classmap[name][scan_id].append(z)

        out_path = label_dir / f"gt_classmap_{min_px}.json"
        with open(out_path, "w") as f:
            json.dump(classmap, f, indent=2)
        print(f"{out_path.name} written")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRI_muscle_2 dataset preprocessing (thigh only)")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/datasets/MRI_muscle_2"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/datasets/MRI_muscle_2/processed"))
    parser.add_argument("--echo", type=str, default="Water", choices=["Water", "Fat", "In_phase", "Opp_phase"])
    parser.add_argument("--crop_size", type=int, default=0,
                        help="Center crop size in-plane (0 = disable). Default 0.")
    args = parser.parse_args()

    preprocess_mri_muscle_2(args.raw_dir, args.out_dir, echo=args.echo, crop_size=args.crop_size)

    echo_dir = args.out_dir / "WATER"
    build_mri_muscle_2_classmap(echo_dir, MRI_MUSCLE_2_LABEL_NAMES, min_pixels_list=[1, 100])
