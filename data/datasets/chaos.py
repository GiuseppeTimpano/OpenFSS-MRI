import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

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

CHAOS_LABEL_NAMES = ["BG", "LIVER", "RK", "LK", "SPLEEN"]
CHAOS_TARGET_SPACING = [1.25, 1.25, 7.70]
CHAOS_CROP_SIZE = 256

# maps CHAOS folder names → (short output name, phase_tag)
# T1DUAL: dcm2niix produces _e1 (OutPhase) + _e2 (InPhase); use InPhase
# T2SPIR: single NIfTI, no suffix
MODALITY_MAP = {
    'T1DUAL': ('T1', '_e2'),
    'T2SPIR': ('T2', ''),
}


def convert_pngs_to_nii(raw_dir: Path, out_dir: Path) -> None:
    """Convert CHAOS PNG ground-truth masks to NIfTI for all modalities."""
    out_dir = Path(out_dir)
    raw_dir = Path(raw_dir)

    for curr_id in os.listdir(raw_dir):
        subject_path = raw_dir / curr_id
        if not subject_path.is_dir():
            continue

        for modality in MODALITY_MAP:  # keys only
            ground_path = subject_path / modality / 'Ground'
            if not ground_path.exists():
                continue

            pngs = sorted(
                glob.glob(str(ground_path / '*.png')),
                key=lambda x: int(os.path.basename(x).split('-')[-1].replace('.png', ''))
            )
            if not pngs:
                continue

            buffer = [np.array(Image.open(fid)) for fid in pngs]
            vol = np.stack(buffer, axis=0)
            vol = np.flip(vol, axis=1).copy()

            src = vol.copy()
            remapped = np.zeros_like(vol)
            for new_val, old_val in enumerate(sorted(np.unique(src))):
                remapped[src == old_val] = new_val
            vol = remapped

            mask_img = sitk.GetImageFromArray(vol.astype(np.uint8))

            mod_out = out_dir / modality
            mod_out.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(mask_img, str(mod_out / f'{curr_id}_{modality}.nii.gz'))


def preprocess_volume(
    masks_dir: Path,
    img_dir: Path,
    out_dir: Path,
    modality: str = 'T2SPIR',
    phase_tag: str = '',
    target_spacing: list = CHAOS_TARGET_SPACING,
    crop_size: int = CHAOS_CROP_SIZE,
) -> None:
    """
    Preprocess one modality: n4 → upper clip → resample → center crop.
    Reads from masks_dir/<modality>/ and img_dir/<modality>/.
    Writes image_*.nii.gz and label_*.nii.gz to out_dir.

    phase_tag: suffix appended to image filename before .nii.gz.
        T1DUAL: dcm2niix produces _e1 (OutPhase) and _e2 (InPhase).
        Pass '_e2' to select InPhase. Empty string for T2SPIR (single file).
    """
    masks_dir = Path(masks_dir) / modality
    img_dir   = Path(img_dir)   / modality
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mask_path in sorted(masks_dir.glob(f'*_{modality}.nii.gz')):
        stem = mask_path.stem.replace('.nii', '')
        subject_id = stem.rsplit('_', 1)[0]

        ref_path = img_dir / f'chaos_{subject_id}_{modality}{phase_tag}.nii.gz'
        if not ref_path.exists():
            print(f'Warning: no image for {mask_path.name} (expected {ref_path.name}), skipping')
            continue

        img_itk = sitk.ReadImage(str(ref_path))
        lbl_itk = sitk.ReadImage(str(mask_path))

        lbl_itk.SetSpacing(img_itk.GetSpacing())
        lbl_itk.SetOrigin(img_itk.GetOrigin())
        lbl_itk.SetDirection(img_itk.GetDirection())

        img_itk = n4_bias_field_correction(img_itk)
        img_itk = intensity_clip_upper(img_itk)
        img_itk = resample_help_function(img_itk, target_spacing, is_label=False)
        img_padval = float(sitk.GetArrayFromImage(img_itk).min())
        img_itk = center_crop_2d(img_itk, crop_size, padval=img_padval)

        lbl_itk = resample_label_onehot(lbl_itk, target_spacing)
        lbl_itk = center_crop_2d(lbl_itk, crop_size, padval=0.0)

        sitk.WriteImage(img_itk, str(out_dir / f'image_{subject_id}.nii.gz'))
        sitk.WriteImage(lbl_itk, str(out_dir / f'label_{subject_id}.nii.gz'))
        print(f'{modality} subject {subject_id} done → {out_dir}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='CHAOS MRI preprocessing pipeline')
    parser.add_argument('--no-supervoxels', action='store_true',
                        help='Skip supervoxel extraction (default: extract)')
    args = parser.parse_args()
    extract_supervoxels = not args.no_supervoxels

    raw_dir  = Path('data/datasets/CHAOS/raw_data/MR')
    tmp_dir  = Path('data/datasets/CHAOS/tmp')
    proc_dir = Path('data/datasets/CHAOS/processed')

    script_path = Path(__file__).parent.parent.parent / 'scripts' / 'data_prep' / 'dcm_to_nii.sh'

    # Step 1: DICOM → NIfTI images (tmp/images/T1DUAL/, tmp/images/T2SPIR/)
    subprocess.run(['bash', str(script_path), str(raw_dir), str(tmp_dir / 'images')], check=True)

    # Step 2: PNG masks → NIfTI (tmp/masks/T1DUAL/, tmp/masks/T2SPIR/)
    convert_pngs_to_nii(raw_dir, tmp_dir / 'masks')

    # Step 3: preprocess each modality → processed/T1/, processed/T2/
    for chaos_name, (short, phase_tag) in MODALITY_MAP.items():
        preprocess_volume(
            masks_dir=tmp_dir / 'masks',
            img_dir=tmp_dir / 'images',
            out_dir=proc_dir / short,
            modality=chaos_name,
            phase_tag=phase_tag,
        )

    # Step 4: GT classmaps per modality
    for short, _ in MODALITY_MAP.values():
        build_gt_classmap(str(proc_dir / short), CHAOS_LABEL_NAMES, [1, 100], str(proc_dir / short))

    # Step 5: supervoxel pseudo-labels per modality
    if extract_supervoxels:
        from utils.supervoxel import run as run_supervoxel, SupervoxelConfig, PRESETS
        sv_cfg = SupervoxelConfig(fg_thresh=PRESETS['CHAOST2']['fg_thresh'])
        for short, _ in MODALITY_MAP.values():
            out = proc_dir / short
            print(f'Extracting supervoxels → {out}')
            run_supervoxel(str(out), str(out), sv_cfg)

    # Step 6: remove intermediate tmp folder
    shutil.rmtree(tmp_dir)
    print('tmp removed. Done.')
