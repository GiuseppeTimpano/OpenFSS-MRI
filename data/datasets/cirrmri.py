"""
CirrMRI600+ (healthy subjects) preprocessing — held-out MRI test set for the
foundation-model comparison. CirrMRI600+ is NOT in the MedSAM2 training list
(published after the MedSAM2 dataset table was frozen), so it serves as a clean
out-of-distribution liver benchmark. Healthy subjects only (no cirrhosis), to keep
the domain shift = scanner/protocol, not pathology.

Input layout (as released on OSF, DOI:10.17605/OSF.IO/CUK24):
  CIRRMR/Healthy_subjects/{T2_W_Healthy,T1_W_Healthy}/{<MOD>_images,<MOD>_masks}/{id}.nii.gz
  masks are binary {0,1}, 1 = liver (already the CHAOS LIVER id).

Output: data/datasets/CIRRMR/processed/<MOD>/{image_*,label_*}.nii.gz
  Same preprocessing + target geometry as CHAOS / AMOS so the domain comparison is
  fair. This set is used as an eval *query target* only (support comes from CHAOS),
  so no supervoxels / gt_classmaps are produced (those are for episodic training).
"""
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
)

# match CHAOS / AMOS so geometry is identical across the comparison
TARGET_SPACING = [1.25, 1.25, 7.70]
CROP_SIZE = 256


def reorient_to_lps(image_itk: sitk.Image) -> sitk.Image:
    """Reorient any NIfTI volume to standard LPS axial orientation."""
    return sitk.DICOMOrient(image_itk, "LPS")


def acquisition_plane(image_itk: sitk.Image) -> str:
    """Native acquisition plane = axis with the largest (slice-thickness) spacing.
    Detected on the RAW image before reorient."""
    sp = np.array(image_itk.GetSpacing())
    return {0: "sagittal", 1: "coronal", 2: "axial"}[int(np.argmax(sp))]


def preprocess_cirrmri(raw_dir: Path, out_dir: Path, modality: str = "T2") -> None:
    img_dir = raw_dir / f"{modality}_W_Healthy" / f"{modality}_images"
    lbl_dir = raw_dir / f"{modality}_W_Healthy" / f"{modality}_masks"
    out_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.nii.gz"),
                       key=lambda p: int(p.name.split('.')[0]))
    print(f"Found {len(img_paths)} CirrMRI healthy {modality} cases")

    kept = 0
    for img_path in img_paths:
        case_id = img_path.name.split('.')[0]            # e.g. "9"
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            print(f"  skip {case_id} — no mask")
            continue

        img_itk = sitk.ReadImage(str(img_path))
        lbl_itk = sitk.ReadImage(str(lbl_path))

        # keep only natively AXIAL acquisitions (CirrMRI is multiplanar) to match
        # CHAOS T2 axial.
        plane = acquisition_plane(img_itk)
        if plane != "axial":
            for stale in out_dir.glob(f"*_{case_id}.nii.gz"):
                stale.unlink()
            print(f"  skip {case_id} — {plane} acquisition (non-axial)")
            continue

        img_itk = reorient_to_lps(img_itk)
        lbl_itk = reorient_to_lps(lbl_itk)

        # mask is already binary {0,1} with 1 = liver = CHAOS LIVER id → no remap

        img_itk = n4_bias_field_correction(img_itk)
        img_itk = intensity_clip_upper(img_itk)
        img_itk = resample_help_function(img_itk, TARGET_SPACING, is_label=False)
        pad_val = float(sitk.GetArrayFromImage(img_itk).min())
        img_itk = center_crop_2d(img_itk, CROP_SIZE, padval=pad_val)

        lbl_itk = resample_label_onehot(lbl_itk, TARGET_SPACING)
        lbl_itk = center_crop_2d(lbl_itk, CROP_SIZE, padval=0.0)

        sitk.WriteImage(img_itk, str(out_dir / f"image_{case_id}.nii.gz"))
        sitk.WriteImage(lbl_itk, str(out_dir / f"label_{case_id}.nii.gz"))
        kept += 1
        print(f"  {case_id} done → {out_dir}")

    print(f"\nKept {kept}/{len(img_paths)} axial {modality} cases → {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["T1", "T2"], default="T2")
    parser.add_argument("--raw_dir", type=str,
                        default="data/datasets/CIRRMR/Healthy_subjects")
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir) if args.out_dir \
        else Path(f"data/datasets/CIRRMR/processed/{args.modality}")

    preprocess_cirrmri(raw_dir, out_dir, args.modality)
    print("Done.")
