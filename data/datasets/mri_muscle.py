"""MRI muscle dataset preprocessing.

Input:  data/datasets/MRI_muscle/raw_data/<subject>/{ImageData/<subject>_<echo>,SegmentationMasks}
Output: data/datasets/MRI_muscle/processed/{WATER,FAT}/{image_<scan>.nii.gz,label_<scan>.nii.gz}

The dataset is a Dixon MRI thigh acquisition. Raw Dixon echoes: WATER and FAT.
Masks are drawn on the FATFRACTION (PDFF) volume.

FATFRACTION is used as the reference grid: labels are merged on FATFRACTION
(native mask space), and all processed echoes (WATER, FAT) are resampled onto
the FATFRACTION grid. This avoids z-offset between echoes that can occur when
using WATER or FAT as reference (P-patient data). Output echoes:
WATER/image_<scan>.nii.gz, FAT/image_<scan>.nii.gz, and labels in each subdir.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.offline_preprocessing import (
    center_crop_2d,
    intensity_clip_upper,
    n4_bias_field_correction,
    z_volume_norm,
)

MRI_MUSCLE_LABEL_NAMES = [
    "BG",
    "L_QF",
    "L_HS",
    "L_SA",
    "L_GR",
    "R_QF",
    "R_HS",
    "R_SA",
    "R_GR",
]

# filename code -> label id
MRI_MUSCLE_LABEL_MAP = {
    "L_QF": 1,
    "L_HS": 2,
    "L_SA": 3,
    "L_GR": 4,
    "R_QF": 5,
    "R_HS": 6,
    "R_SA": 7,
    "R_GR": 8,
}

# Subjects known to have no masks or bad data.
EXCLUDED_SUBJECTS = {"HV009_1"}


def discover_subjects(root: Path) -> list[str]:
    """Return subject IDs that have at least one .mha mask."""
    root = Path(root)
    subjects = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        sid = path.name
        if sid in EXCLUDED_SUBJECTS:
            continue
        mask_dir = path / "SegmentationMasks"
        if mask_dir.exists() and any(mask_dir.glob("*.mha")):
            subjects.append(sid)
    return subjects


def discover_stacks(subject_dir: Path) -> list[str]:
    """Return stack names (e.g. stack1, stack2, stack3) found in masks."""
    mask_dir = Path(subject_dir) / "SegmentationMasks"
    stacks = set()
    pattern = re.compile(rf"^{re.escape(Path(subject_dir).name)}_(stack\d+)_[LR]_[A-Z]+\.mha$")
    for p in mask_dir.glob("*.mha"):
        m = pattern.match(p.name)
        if m:
            stacks.add(m.group(1))
    return sorted(stacks)


def _find_echo_files(subject_dir: Path, echo: str, stack: str) -> list[Path]:
    """Case-insensitive search for files of one echo/stack, best order first."""
    subject_dir = Path(subject_dir)
    sid = subject_dir.name
    echo_dir = subject_dir / "ImageData" / f"{sid}_{echo}"
    if not echo_dir.exists():
        return []
    target = f"{sid}_{echo}_{stack}".lower()
    found = []
    for p in echo_dir.iterdir():
        name = p.name.lower()
        if name.startswith(target) and name[len(target):] in (".nii.gz", ".nii", ".dcm"):
            found.append(p)
    order = {".nii.gz": 0, ".nii": 1, ".dcm": 2}
    found.sort(key=lambda p: order[p.suffix.lower() if p.suffix.lower() != ".gz" else ".nii.gz"])
    return found


def _find_nii_anchor(subject_dir: Path, stack: str) -> sitk.Image | None:
    """Return the first echo NIfTI (FAT/FATFRACTION/WATER) with sane geometry."""
    subject_dir = Path(subject_dir)
    for echo in ["FAT", "FATFRACTION", "WATER"]:
        for p in _find_echo_files(subject_dir, echo, stack):
            if p.suffix.lower() not in (".nii", ".gz"):
                continue
            img = sitk.ReadImage(str(p))
            if any(abs(v) > 1e-3 for v in img.GetOrigin()):
                return img
    return None


def load_echo(subject_dir: Path, stack: str, echo: str) -> sitk.Image | None:
    """Load one echo volume for a subject/stack. Returns None if missing.

    DICOM fallback (e.g. HV012-015 WATER): when only a .dcm is available its
    geometry is often bogus (origin 0, z-spacing 1); we fix it by copying
    geometry from a sibling echo NIfTI of the same stack.
    """
    subject_dir = Path(subject_dir)
    files = _find_echo_files(subject_dir, echo, stack)
    if not files:
        return None
    img = sitk.ReadImage(str(files[0]))
    if files[0].suffix.lower() == ".dcm":
        anchor = _find_nii_anchor(subject_dir, stack)
        if anchor is not None and anchor.GetSize() == img.GetSize():
            img.CopyInformation(anchor)
    return img


def merge_masks(
    subject_dir: Path,
    stack: str,
    reference: sitk.Image,
    label_map: dict[str, int] | None = None,
) -> sitk.Image:
    """Merge per-muscle .mha masks into one multi-label volume in reference grid."""
    subject_dir = Path(subject_dir)
    sid = subject_dir.name
    mask_dir = subject_dir / "SegmentationMasks"
    label_map = MRI_MUSCLE_LABEL_MAP if label_map is None else label_map

    ref_arr = sitk.GetArrayFromImage(reference)
    out = np.zeros_like(ref_arr, dtype=np.uint8)

    pattern = re.compile(rf"^{re.escape(sid)}_{re.escape(stack)}_([LR]_[A-Z]+)\.mha$")
    for p in sorted(mask_dir.glob("*.mha")):
        m = pattern.match(p.name)
        if not m:
            continue
        code = m.group(1)
        label = label_map.get(code)
        if label is None:
            continue

        m_img = sitk.ReadImage(str(p))
        m_resampled = sitk.Resample(
            m_img,
            reference,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0.0,
            m_img.GetPixelID(),
        )
        m_arr = sitk.GetArrayFromImage(m_resampled)
        out[m_arr > 0] = label

    lbl = sitk.GetImageFromArray(out)
    lbl.CopyInformation(reference)
    return lbl


def resample_to_reference(
    image: sitk.Image,
    reference: sitk.Image,
    is_label: bool = False,
) -> sitk.Image:
    """Resample `image` onto the `reference` grid (physical-space alignment).

    When size/spacing/direction already match the reference, the images are
    index-for-index aligned and only origin differs — for this dataset that
    origin difference is a per-echo metadata bug (P patients), not a real
    physical shift, so we trust the slice index and copy the reference's
    geometry directly instead of resampling against the bogus origin.
    """
    if (
        image.GetSize() == reference.GetSize()
        and image.GetSpacing() == reference.GetSpacing()
        and image.GetDirection() == reference.GetDirection()
    ):
        out = sitk.Image(image)
        out.CopyInformation(reference)
        return out
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        image,
        reference,
        sitk.Transform(),
        interp,
        0.0,
        image.GetPixelID(),
    )


def preprocess_mri_muscle(
    raw_dir: Path,
    out_dir: Path,
    echoes: list[str] | None = None,
    reference_echo: str = "WATER",
    stacks: list[str] | None = None,
    crop_size: int = 0,
) -> None:
    """Convert raw MRI_muscle layout into standard image_/label_ NIfTI files.

    Each subject x stack becomes one scan. FATFRACTION is used as the reference
    grid (labels are drawn on FATFRACTION); all processed echoes (WATER, FAT)
    are resampled onto the FATFRACTION grid. Falls back to `reference_echo` when
    FATFRACTION is missing.
    """
    echoes = ["WATER", "FAT"] if echoes is None else echoes
    if reference_echo not in echoes:
        echoes = [reference_echo, *echoes]

    raw_dir = Path(raw_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    for sid in discover_subjects(raw_dir):
        subject_dir = raw_dir / sid
        avail_stacks = discover_stacks(subject_dir)
        sel = avail_stacks if stacks is None else [s for s in stacks if s in avail_stacks]
        for stack in sel:
            # Use FATFRACTION as reference grid: masks are drawn on FATFRACTION,
            # so labels stay in their native space and avoid z-offset (P patients).
            ref = load_echo(subject_dir, stack, "FATFRACTION")
            if ref is None:
                ref = load_echo(subject_dir, stack, reference_echo)
            if ref is None:
                print(f"  skip {sid} {stack} — no reference echo available")
                continue

            lbl = merge_masks(subject_dir, stack, ref)
            scan = f"{sid}_{stack}"

            # preprocess echoes: N4 → clip 99.5% → resample → center crop → z-score
            for echo in echoes:
                img = load_echo(subject_dir, stack, echo)
                if img is None:
                    print(f"  skip {sid} {stack} echo {echo} — missing")
                    continue
                img = n4_bias_field_correction(img)
                img = intensity_clip_upper(img)
                if echo != "FATFRACTION":
                    img = resample_to_reference(img, ref, is_label=False)
                if crop_size > 0:
                    padval = float(sitk.GetArrayFromImage(img).min())
                    img = center_crop_2d(img, crop_size, padval=padval)
                img = z_volume_norm(img)
                save_path = out_dir / echo
                save_path.mkdir(parents=True, exist_ok=True)
                out_img = save_path / f"image_{scan}.nii.gz"
                sitk.WriteImage(img, str(out_img))

            # crop label to match
            if crop_size > 0:
                lbl = center_crop_2d(lbl, crop_size, padval=0.0)
            for echo in echoes:
                save_path = out_dir / echo
                save_path.mkdir(parents=True, exist_ok=True)
                out_lbl = save_path / f"label_{scan}.nii.gz"
                sitk.WriteImage(lbl, str(out_lbl))

            print(f"  {scan} -> {out_dir}/{{WATER,FAT}} (FATFRACTION grid)")


def build_mri_muscle_classmap(
    label_dir: Path,
    label_names: list[str],
    min_pixels_list: list[int],
) -> None:
    """Build gt_classmap JSON files using the full scan stem as key.

    Unlike the generic build_gt_classmap, scan IDs here are strings like
    'HV001_1_stack1' rather than numeric suffixes.
    """
    label_dir = Path(label_dir)
    label_paths = sorted(label_dir.glob("label_*.nii.gz"))
    if not label_paths:
        raise FileNotFoundError(f"No label_*.nii.gz in {label_dir}")

    for min_px in min_pixels_list:
        classmap = {name: {} for name in label_names}

        for lbl_path in label_paths:
            scan_id = lbl_path.stem.replace(".nii", "").replace("label_", "")
            lbl_vol = sitk.GetArrayFromImage(sitk.ReadImage(str(lbl_path)))
            slice_sum = int(np.sum(lbl_vol))

            for cls_idx, name in enumerate(label_names):
                classmap[name][scan_id] = []

            for z in range(lbl_vol.shape[0]):
                slc = lbl_vol[z]
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

    parser = argparse.ArgumentParser(description="MRI muscle dataset preprocessing")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/datasets/MRI_muscle/raw_data"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/datasets/MRI_muscle/processed"))
    parser.add_argument("--echoes", type=str, nargs="+", default=["WATER", "FAT"],
                        choices=["WATER", "FAT", "FATFRACTION"])
    parser.add_argument("--reference_echo", type=str, default="WATER",
                        choices=["WATER", "FAT", "FATFRACTION"])
    parser.add_argument("--stacks", type=str, nargs="+", default=None,
                        help="Stacks to process (default: all). e.g. stack2 stack3")
    parser.add_argument("--crop_size", type=int, default=0,
                        help="Center crop size in-plane (0 = disable). Default 0.")
    args = parser.parse_args()

    preprocess_mri_muscle(
        args.raw_dir,
        args.out_dir,
        echoes=args.echoes,
        reference_echo=args.reference_echo,
        stacks=args.stacks,
        crop_size=args.crop_size,
    )
    for echo in args.echoes:
        echo_dir = args.out_dir / echo
        echo_dir.mkdir(parents=True, exist_ok=True)
        build_mri_muscle_classmap(
            echo_dir,
            MRI_MUSCLE_LABEL_NAMES,
            min_pixels_list=[1, 100],
        )
