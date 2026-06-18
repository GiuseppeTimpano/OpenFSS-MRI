"""
AMOS22 MRI preprocessing pipeline.

Input:  data/datasets/AMOS/raw_data/{images,labels}/amos_05xx.nii.gz
Output: data/datasets/AMOS/processed/T2/{image_*,label_*}.nii.gz
        + gt_classmap_*.json + supervoxel_*.nii.gz

Label remapping  AMOS → CHAOS convention:
  AMOS: 0=BG, 1=spleen, 2=RK, 3=LK, 4=gallbladder, 5=esophagus,
        6=liver, 7=stomach, 8=aorta, 9=postcava, 10=pancreas, ...
  CHAOS: 0=BG, 1=LIVER, 2=RK, 3=LK, 4=SPLEEN
"""
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from data.offline_preprocessing import (
    n4_bias_field_correction,
    intensity_clip_upper,
    resample_help_function,
    resample_label_onehot,
    center_crop_2d,
    build_gt_classmap,
)

AMOS_LABEL_NAMES = ["BG", "LIVER", "RK", "LK", "SPLEEN"]
# same as CHAOS so supervoxel thresholds transfer
TARGET_SPACING = [1.25, 1.25, 7.70]
CROP_SIZE = 256

# AMOS label id → CHAOS label id (0 = discard)
AMOS_TO_CHAOS = {
    0: 0,   # BG
    6: 1,   # liver
    2: 2,   # right kidney
    3: 3,   # left kidney
    1: 4,   # spleen
}


def remap_labels(lbl_itk: sitk.Image) -> sitk.Image:
    arr = sitk.GetArrayFromImage(lbl_itk).astype(np.int32)
    out = np.zeros_like(arr)
    for amos_id, chaos_id in AMOS_TO_CHAOS.items():
        if chaos_id > 0:
            out[arr == amos_id] = chaos_id
    result = sitk.GetImageFromArray(out)
    result.CopyInformation(lbl_itk)
    return result


def reorient_to_lps(image_itk: sitk.Image) -> sitk.Image:
    """Reorient any NIfTI volume to standard LPS axial orientation."""
    return sitk.DICOMOrient(image_itk, "LPS")


def preprocess_amos(raw_dir: Path, out_dir: Path) -> None:
    img_dir = raw_dir / "images"
    lbl_dir = raw_dir / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("amos_*.nii.gz"))
    print(f"Found {len(img_paths)} AMOS MRI cases")

    for img_path in img_paths:
        case_id = img_path.stem.replace(".nii", "")  # e.g. amos_0507
        lbl_path = lbl_dir / img_path.name

        if not lbl_path.exists():
            print(f"  skip {case_id} — no label")
            continue

        img_itk = sitk.ReadImage(str(img_path))
        lbl_itk = sitk.ReadImage(str(lbl_path))

        # reorient both to LPS axial
        img_itk = reorient_to_lps(img_itk)
        lbl_itk = reorient_to_lps(lbl_itk)

        # remap AMOS labels → CHAOS convention
        lbl_itk = remap_labels(lbl_itk)

        # preprocess image
        img_itk = n4_bias_field_correction(img_itk)
        img_itk = intensity_clip_upper(img_itk)
        img_itk = resample_help_function(img_itk, TARGET_SPACING, is_label=False)
        pad_val = float(sitk.GetArrayFromImage(img_itk).min())
        img_itk = center_crop_2d(img_itk, CROP_SIZE, padval=pad_val)

        # preprocess label
        lbl_itk = resample_label_onehot(lbl_itk, TARGET_SPACING)
        lbl_itk = center_crop_2d(lbl_itk, CROP_SIZE, padval=0.0)

        out_img = out_dir / f"image_{case_id}.nii.gz"
        out_lbl = out_dir / f"label_{case_id}.nii.gz"
        sitk.WriteImage(img_itk, str(out_img))
        sitk.WriteImage(lbl_itk, str(out_lbl))
        print(f"  {case_id} done → {out_dir}")


if __name__ == "__main__":
    raw_dir  = Path("data/datasets/AMOS/raw_data")
    proc_dir = Path("data/datasets/AMOS/processed/T2")

    preprocess_amos(raw_dir, proc_dir)

    build_gt_classmap(str(proc_dir), AMOS_LABEL_NAMES, [1, 100], str(proc_dir))
    print("gt_classmaps written")

    from utils.supervoxel import run as run_supervoxel, SupervoxelConfig, PRESETS
    sv_cfg = SupervoxelConfig(fg_thresh=PRESETS["CHAOST2"]["fg_thresh"])
    print(f"Extracting supervoxels → {proc_dir}")
    run_supervoxel(str(proc_dir), str(proc_dir), sv_cfg)
    print("Done.")
