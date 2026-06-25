"""
Unpaired 2D slice dataset for CycleGAN training on NIfTI MRI volumes.
Loads foreground slices from two separate domain directories.
Normalization: per-slice percentile clip (5-99.5) + min-max → [0,1].
"""

import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Percentile clip (5-99.5th) + min-max → [0, 1] float32."""
    lo = np.percentile(arr, 5)
    hi = np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)
    mn, mx = arr.min(), arr.max()
    return ((arr - mn) / (mx - mn + 1e-8)).astype(np.float32)


class UnpairedNIfTIDataset(Dataset):
    """
    Unpaired dataset: returns (slice_A, slice_B) where A and B are drawn
    independently from two domain directories. Length = max(|A|, |B|).

    dir_A, dir_B : paths containing image_*.nii.gz
                   and optionally fgmask_*.nii.gz for foreground selection.
    use_fgmask   : if True and fgmask exists, use it to select FG slices.
                   Fallback: slices with >5% nonzero pixels.
    """

    def __init__(self, dir_A: str, dir_B: str, use_fgmask: bool = True):
        self.slices_A = self._load_slices(dir_A, use_fgmask)
        self.slices_B = self._load_slices(dir_B, use_fgmask)
        if not self.slices_A:
            raise ValueError(f'No foreground slices found in {dir_A}')
        if not self.slices_B:
            raise ValueError(f'No foreground slices found in {dir_B}')
        print(f'Dataset: {len(self.slices_A)} slices A, {len(self.slices_B)} slices B')

    def _load_slices(self, data_dir: str, use_fgmask: bool) -> list[np.ndarray]:
        slices = []
        paths = sorted(glob.glob(os.path.join(data_dir, 'image_*.nii.gz')))
        for img_path in paths:
            sid = os.path.basename(img_path).replace('image_', '').replace('.nii.gz', '')
            vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)

            fg_path = os.path.join(data_dir, f'fgmask_{sid}.nii.gz')
            if use_fgmask and os.path.exists(fg_path):
                fg  = sitk.GetArrayFromImage(sitk.ReadImage(fg_path))
                idx = np.where(fg.any(axis=(1, 2)))[0]
            else:
                # Slices with >5% nonzero voxels
                idx = np.where(
                    (vol > vol.min()).mean(axis=(1, 2)) > 0.05
                )[0]

            for z in idx:
                slices.append(vol[z].copy())
        return slices

    def __len__(self) -> int:
        return max(len(self.slices_A), len(self.slices_B))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        a = normalize_slice(self.slices_A[idx % len(self.slices_A)])
        b = normalize_slice(self.slices_B[np.random.randint(len(self.slices_B))])
        return torch.from_numpy(a).unsqueeze(0), torch.from_numpy(b).unsqueeze(0)
