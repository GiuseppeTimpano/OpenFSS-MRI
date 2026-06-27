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

Organ labels (for region-aware loss):
  If label_{sid}.nii.gz exists, per-slice organ masks are loaded alongside the
  image. CHAOS encoding: 1=liver, 2=right kidney, 3=left kidney, 4=spleen.
  Body soft-tissue NOT belonging to an organ (fgmask & label==0) is relabeled
  class 5 ('rest'), so {1..5} partitions the whole body; air stays 0 (ignored).
  Per-domain organ intensity stats (mean/std in [-1,1] space) are precomputed
  in __init__ and exposed as self.stats_A / self.stats_B for the region loss.
  Missing label file → all-zero mask for that domain (loss contributes nothing).
"""

import glob
import json
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
                 aug_translate: float = 0.05, aug_scale: float = 0.05,
                 case_ids_A: list[str] | None = None,
                 case_ids_B: list[str] | None = None):
        self.min_body = min_body
        self.augment = augment
        self.aug_degrees = aug_degrees
        self.aug_translate = aug_translate
        self.aug_scale = aug_scale
        self.slices_A, self.subj_A, self.depth_A, self.labels_A = self._load_slices(dir_A, use_fgmask, case_ids_A)
        self.slices_B, self.subj_B, self.depth_B, self.labels_B = self._load_slices(dir_B, use_fgmask, case_ids_B)
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

        # per-domain organ intensity stats (mean/std in [-1,1]) for region-aware loss
        self.stats_A = self._compute_stats(self.slices_A, self.labels_A)
        self.stats_B = self._compute_stats(self.slices_B, self.labels_B)
        print(f'Organ stats A: { {k: (round(m, 2), round(s, 2)) for k, (m, s) in self.stats_A.items()} }')
        print(f'Organ stats B: { {k: (round(m, 2), round(s, 2)) for k, (m, s) in self.stats_B.items()} }')

    # CHAOS organ labels + 'rest' body class (see module docstring)
    ORGAN_CLASSES = (1, 2, 3, 4, 5)

    @staticmethod
    def case_ids_from_manifest(manifest_path: str, manufacturer: str | None = None,
                               model: str | None = None) -> list[str]:
        """Return case_ids from scanner_manifest.json filtered by manufacturer and/or model.
        Matching is case-insensitive substring."""
        with open(manifest_path) as f:
            manifest = json.load(f)
        ids = []
        for cid, m in manifest.items():
            if manufacturer and manufacturer.lower() not in m.get("manufacturer", "").lower():
                continue
            if model and model.lower() not in m.get("model", "").lower():
                continue
            ids.append(cid)
        return sorted(ids)

    def _load_slices(self, data_dir: str, use_fgmask: bool,
                     case_ids: list[str] | None = None):
        slices: list[np.ndarray] = []
        subjects: list[str] = []
        depths: list[float] = []
        labels: list[np.ndarray] = []
        paths = sorted(glob.glob(os.path.join(data_dir, 'image_*.nii.gz')))
        if case_ids is not None:
            case_ids_set = set(case_ids)
            paths = [p for p in paths
                     if os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                     in case_ids_set]
        for img_path in paths:
            sid = os.path.basename(img_path).replace('image_', '').replace('.nii.gz', '')
            vol = sitk.GetArrayFromImage(sitk.ReadImage(img_path)).astype(np.float32)

            fg_path  = os.path.join(data_dir, f'fgmask_{sid}.nii.gz')
            lab_path = os.path.join(data_dir, f'label_{sid}.nii.gz')
            fgvol  = sitk.GetArrayFromImage(sitk.ReadImage(fg_path))  if os.path.exists(fg_path)  else None
            labvol = sitk.GetArrayFromImage(sitk.ReadImage(lab_path)) if os.path.exists(lab_path) else None

            if use_fgmask and fgvol is not None:
                idx = np.where(fgvol.any(axis=(1, 2)))[0]
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
                # organ label slice; relabel body soft-tissue (in fg, no organ) as class 5
                if labvol is not None:
                    ls = labvol[z].astype(np.float32).copy()
                    body = fgvol[z] > 0 if fgvol is not None else (s > s.min())
                    ls[(ls == 0) & body] = 5.0
                else:
                    ls = np.zeros_like(s, dtype=np.float32)
                slices.append(s.copy())
                labels.append(ls)
                subjects.append(sid)
                depths.append((int(z) - z0) / span)  # 0 (top FG) .. 1 (bottom FG)
        return slices, subjects, depths, labels

    def _compute_stats(self, slices, labels) -> dict[int, tuple[float, float]]:
        """Per-organ (mean, std) over all FG slices in normalize_slice [-1,1] space."""
        acc = {k: [0.0, 0.0, 0] for k in self.ORGAN_CLASSES}  # sum, sumsq, count
        for s, lab in zip(slices, labels):
            a = normalize_slice(s)
            for k in self.ORGAN_CLASSES:
                m = lab == k
                n = int(m.sum())
                if n:
                    v = a[m]
                    acc[k][0] += float(v.sum())
                    acc[k][1] += float((v * v).sum())
                    acc[k][2] += n
        stats: dict[int, tuple[float, float]] = {}
        for k, (sm, ss, n) in acc.items():
            if n > 0:
                mean = sm / n
                std = max(ss / n - mean * mean, 0.0) ** 0.5
                stats[k] = (mean, std)
        return stats

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

    def _augment(self, t: torch.Tensor, lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """hflip + small affine applied IDENTICALLY to image and label (label uses
        NEAREST to keep class ids intact). Fill: image=background, label=0 (air)."""
        fill = float(t.min())
        if random.random() < 0.5:
            t = TF.hflip(t)
            lab = TF.hflip(lab)
        angle = random.uniform(-self.aug_degrees, self.aug_degrees)
        tx = random.uniform(-self.aug_translate, self.aug_translate) * t.shape[-1]
        ty = random.uniform(-self.aug_translate, self.aug_translate) * t.shape[-2]
        scale = random.uniform(1.0 - self.aug_scale, 1.0 + self.aug_scale)
        t = TF.affine(t, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                      interpolation=InterpolationMode.BILINEAR, fill=fill)
        lab = TF.affine(lab, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                        interpolation=InterpolationMode.NEAREST, fill=0.0)
        return t, lab

    def __len__(self) -> int:
        return max(len(self.slices_A), len(self.slices_B))

    def __getitem__(self, idx: int):
        ia = idx % len(self.slices_A)
        a = normalize_slice(self.slices_A[ia])
        jb = self._sample_B(self.subj_A[ia], self.depth_A[ia])
        b = normalize_slice(self.slices_B[jb])
        ta = torch.from_numpy(a).unsqueeze(0)
        tb = torch.from_numpy(b).unsqueeze(0)
        la = torch.from_numpy(self.labels_A[ia]).unsqueeze(0)
        lb = torch.from_numpy(self.labels_B[jb]).unsqueeze(0)
        if self.augment:
            ta, la = self._augment(ta, la)
            tb, lb = self._augment(tb, lb)
        return ta, tb, la, lb
