"""MRI_muscle_2 dataset preprocessing (external figshare thigh/calf dataset) -- Calf region.

Input:  data/datasets/MRI_muscle_2/<subject>/Calf/{Water.nii.gz, mask_muscles.nii.gz}
Output: data/datasets/MRI_muscle_2/processed_calf/WATER/{image_<scan>.nii.gz, label_<scan>.nii.gz}

Mirrors data/datasets/mri_muscle_2.py (Thigh preprocessing), same crop/N4/norm
pipeline. Difference: mask_muscles here has 9 discrete labels (1-9) with no
anatomical name mapping shipped with the dataset (no README/table under
data/datasets/MRI_muscle_2/), so no grouping is applied -- each raw id is kept
as its own class, named CALF1..CALF9 generically.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.offline_preprocessing import (
    center_crop_2d,
    crop_to_label_bbox_2d,
    intensity_clip_upper,
    n4_bias_field_correction,
    z_volume_norm,
)

MRI_MUSCLE_2_CALF_LABEL_NAMES = [
    "BG", "CALF1", "CALF2", "CALF3", "CALF4", "CALF5", "CALF6", "CALF7", "CALF8", "CALF9",
]


def discover_subjects(raw_dir: Path) -> list[str]:
    """Return subject IDs (e.g. '01') that have a Calf/mask_muscles.nii.gz."""
    raw_dir = Path(raw_dir)
    subjects = []
    for path in sorted(raw_dir.iterdir()):
        if not path.is_dir():
            continue
        if (path / "Calf" / "mask_muscles.nii.gz").exists() and (path / "Calf" / "Water.nii.gz").exists():
            subjects.append(path.name)
    return subjects


def preprocess_mri_muscle_2_calf(
    raw_dir: Path,
    out_dir: Path,
    echo: str = "Water",
    crop_size: int = 0,
    leg_margin_px: int = 40,
) -> None:
    """Convert raw MRI_muscle_2 Calf layout into standard image_/label_ NIfTI files."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    echo_out_dir = out_dir / "WATER"
    echo_out_dir.mkdir(parents=True, exist_ok=True)

    for sid in discover_subjects(raw_dir):
        calf_dir = raw_dir / sid / "Calf"
        img = sitk.ReadImage(str(calf_dir / f"{echo}.nii.gz"))
        lbl = sitk.ReadImage(str(calf_dir / "mask_muscles.nii.gz"))
        whole_muscle_sat = sitk.ReadImage(str(calf_dir / "mask_whole_muscle_SAT.nii.gz"))

        # same rationale as Thigh: mask_muscles is discrete muscle blobs with gaps at
        # boundaries, mask_whole_muscle_SAT is a filled muscle+fat silhouette -- tighter,
        # more robust crop bound for the single annotated leg.
        img, lbl = crop_to_label_bbox_2d(img, lbl, bbox_source_itk=whole_muscle_sat,
                                         margin_px=leg_margin_px)

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


def build_mri_muscle_2_calf_classmap(
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

    parser = argparse.ArgumentParser(description="MRI_muscle_2 dataset preprocessing (calf only)")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/datasets/MRI_muscle_2"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/datasets/MRI_muscle_2/processed_calf"))
    parser.add_argument("--echo", type=str, default="Water", choices=["Water", "Fat", "In_phase", "Opp_phase"])
    parser.add_argument("--crop_size", type=int, default=0,
                        help="Center crop size in-plane (0 = disable). Default 0.")
    parser.add_argument("--leg_margin_px", type=int, default=40,
                        help="Margin (px) around the annotated leg's GT bbox when cropping "
                             "out the other, unannotated leg. Default 40.")
    args = parser.parse_args()

    preprocess_mri_muscle_2_calf(args.raw_dir, args.out_dir, echo=args.echo, crop_size=args.crop_size,
                                 leg_margin_px=args.leg_margin_px)

    echo_dir = args.out_dir / "WATER"
    build_mri_muscle_2_calf_classmap(echo_dir, MRI_MUSCLE_2_CALF_LABEL_NAMES, min_pixels_list=[1, 100])
