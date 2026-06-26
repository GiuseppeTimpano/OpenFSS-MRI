"""
2D slice dataset for CycleGAN training on NIfTI MRI volumes.
Loads foreground slices from two domain directories.
Normalization: per-slice percentile clip (5-99.5) + min-max → [-1,1] (tanh range).

Pairing (pair_mode):
  'subject' : A slice ↔ B slice of the SAME subject_id at nearest anatomical
              depth. Exploits the slight T1↔T2 co-registration (CHAOS T2→T1).
  'depth'   : A slice ↔ B slice at nearest normalized depth, across all subjects
              (anatomical-level proxy for unpaired domains, e.g. AMOS→CHAOS).
  'random'  : original unpaired sampling (B drawn uniformly at random).
  'auto'    : 'subject' if subject_ids overlap between A and B, else 'depth'.
A tolerance window keeps several B candidates per query → some variety is
retained (CycleGAN robustness) while enforcing rough anatomical correspondence.
"""

import glob
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset


def normalize_slice(arr: np.ndarray) -> np.ndarray:
    """Percentile clip (5-99.5th) + min-max → [-1, 1] float32 (matches generator tanh)."""
    lo = np.percentile(arr, 5)
    hi = np.percentile(arr, 99.5)
    arr = np.clip(arr, lo, hi)
    mn, mx = arr.min(), arr.max()
    arr01 = (arr - mn) / (mx - mn + 1e-8)
    return (arr01 * 2.0 - 1.0).astype(np.float32)


class UnpairedNIfTIDataset(Dataset):
    """
    Returns (slice_A, slice_B). A is iterated in order; B is chosen per
    pair_mode (see module docstring). Length = max(|A|, |B|).

    dir_A, dir_B : paths containing image_*.nii.gz
                   and optionally fgmask_*.nii.gz for foreground selection.
    use_fgmask   : if True and fgmask exists, use it to select FG slices.
                   Fallback: slices with >5% nonzero pixels.
    min_body     : drop slices whose body fraction < this even if fgmask flags
                   them FG. Removes degenerate near-black slices (real_B black).
    pair_mode    : 'auto' | 'subject' | 'depth' | 'random'.
    depth_tol    : half-width of the depth window for candidate B slices,
                   as a fraction of volume depth (0..1).
    augment      : per-slice train-time augmentation (hflip + small affine),
                   applied independently to A and B. Off for deterministic use.
    """

    def __init__(self, dir_A: str, dir_B: str, use_fgmask: bool = True,
                 min_body: float = 0.05, pair_mode: str = 'auto', depth_tol: float = 0.1,
                 augment: bool = True, aug_degrees: float = 10.0,
                 aug_translate: float = 0.05, aug_scale: float = 0.05):
        self.min_body = min_body
        self.augment = augment
        self.aug_degrees = aug_degrees
        self.aug_translate = aug_translate
        self.aug_scale = aug_scale
        self.slices_A, self.subj_A, self.depth_A = self._load_slices(dir_A, use_fgmask)
        self.slices_B, self.subj_B, self.depth_B = self._load_slices(dir_B, use_fgmask)
        if not self.slices_A:
            raise ValueError(f'No foreground slices found in {dir_A}')
        if not self.slices_B:
            raise ValueError(f'No foreground slices found in {dir_B}')

        self.depth_tol = depth_tol
        self.depth_B = np.asarray(self.depth_B, dtype=np.float32)

        # resolve 'auto'
        if pair_mode == 'auto':
            overlap = set(self.subj_A) & set(self.subj_B)
            pair_mode = 'subject' if overlap else 'depth'
        self.pair_mode = pair_mode

        # index B by subject for fast same-subject lookup
        self.b_by_subject: dict[str, np.ndarray] = {}
        if pair_mode == 'subject':
            for j, s in enumerate(self.subj_B):
                self.b_by_subject.setdefault(s, []).append(j)
            self.b_by_subject = {s: np.asarray(v) for s, v in self.b_by_subject.items()}

        n_pairable = (len(set(self.subj_A) & set(self.subj_B))
                      if pair_mode == 'subject' else len(set(self.subj_B)))
        print(f'Dataset: {len(self.slices_A)} slices A, {len(self.slices_B)} slices B '
              f'| pair_mode={pair_mode} (tol={depth_tol}) | {n_pairable} subjects pairable')

    def _load_slices(self, data_dir: str, use_fgmask: bool):
        slices: list[np.ndarray] = []
        subjects: list[str] = []
        depths: list[float] = []
        paths = sorted(glob.glob(os.path.join(data_dir, 'image_*.nii.gz')))
        for img_path in paths:
            sid = os.path.basename(img_path).replace('image_', '').replace('.nii.gz', '')
            vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)

            fg_path = os.path.join(data_dir, f'fgmask_{sid}.nii.gz')
            if use_fgmask and os.path.exists(fg_path):
                fg  = sitk.GetArrayFromImage(sitk.ReadImage(fg_path))
                idx = np.where(fg.any(axis=(1, 2)))[0]
            else:
                idx = np.where((vol > vol.min()).mean(axis=(1, 2)) > 0.05)[0]

            if len(idx) == 0:
                continue
            z0, z1 = int(idx.min()), int(idx.max())
            span = max(1, z1 - z0)
            for z in idx:
                s = vol[z]
                # drop degenerate near-black slices (fgmask can flag empty z)
                if (s > s.min()).mean() < self.min_body:
                    continue
                slices.append(s.copy())
                subjects.append(sid)
                depths.append((int(z) - z0) / span)  # 0 (top FG) .. 1 (bottom FG)
        return slices, subjects, depths

    def _sample_B(self, subj: str, depth: float) -> int:
        """Pick a B slice index per pair_mode, within the depth window."""
        if self.pair_mode == 'random':
            return np.random.randint(len(self.slices_B))

        if self.pair_mode == 'subject' and subj in self.b_by_subject:
            cand = self.b_by_subject[subj]
            d = self.depth_B[cand]
            within = cand[np.abs(d - depth) <= self.depth_tol]
            pool = within if len(within) else cand[[np.argmin(np.abs(d - depth))]]
            return int(np.random.choice(pool))

        # depth mode (or subject fallback when subj absent in B)
        diff = np.abs(self.depth_B - depth)
        within = np.where(diff <= self.depth_tol)[0]
        pool = within if len(within) else np.array([int(np.argmin(diff))])
        return int(np.random.choice(pool))

    def _augment(self, t: torch.Tensor) -> torch.Tensor:
        """hflip + small affine (rotation/translation/scale). Fill = background."""
        fill = float(t.min())
        if random.random() < 0.5:
            t = TF.hflip(t)
        angle = random.uniform(-self.aug_degrees, self.aug_degrees)
        tx = random.uniform(-self.aug_translate, self.aug_translate) * t.shape[-1]
        ty = random.uniform(-self.aug_translate, self.aug_translate) * t.shape[-2]
        scale = random.uniform(1.0 - self.aug_scale, 1.0 + self.aug_scale)
        return TF.affine(t, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                         interpolation=InterpolationMode.BILINEAR, fill=fill)

    def __len__(self) -> int:
        return max(len(self.slices_A), len(self.slices_B))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ia = idx % len(self.slices_A)
        a = normalize_slice(self.slices_A[ia])
        jb = self._sample_B(self.subj_A[ia], self.depth_A[ia])
        b = normalize_slice(self.slices_B[jb])
        ta = torch.from_numpy(a).unsqueeze(0)
        tb = torch.from_numpy(b).unsqueeze(0)
        if self.augment:
            ta = self._augment(ta)
            tb = self._augment(tb)
        return ta, tb
