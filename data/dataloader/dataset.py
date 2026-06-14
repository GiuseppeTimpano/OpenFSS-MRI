import glob
import json
import os
import random
import re
from typing import Optional

import numpy as np
import SimpleITK as sitk
import torch

from monai.data.dataset import Dataset
from data.augmentation import get_train_transform


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def get_fold_ids(
    data_dir: str,
    fold: int,
    n_folds: int = 4,
) -> tuple[list[str], list[str]]:
    """
    Split scan IDs found in data_dir into (train_ids, test_ids) for the given fold.
    IDs are sorted numerically, divided into n_folds chunks; chunk at `fold` is test.
    """
    paths = sorted(
        glob.glob(os.path.join(data_dir, 'image_*.nii.gz')),
        key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[-1]),
    )
    all_ids = [re.findall(r'\d+', os.path.basename(p))[-1] for p in paths]

    chunk = len(all_ids) // n_folds
    chunks = [all_ids[i * chunk:(i + 1) * chunk] for i in range(n_folds)]
    for j, extra in enumerate(all_ids[n_folds * chunk:]):
        chunks[j].append(extra)

    test_ids  = chunks[fold]
    train_ids = [sid for i, c in enumerate(chunks) if i != fold for sid in c]
    return train_ids, test_ids


class _ScanView:
    """Slice-indexable view into one scan loaded in RAM."""

    def __init__(self, img: np.ndarray, lbl: np.ndarray, sv: Optional[np.ndarray]):
        self._img = img   # [D, H, W] float32, already normalized
        self._lbl = lbl   # [D, H, W] int32
        self._sv  = sv    # [D, H, W] int32 | None

    def __getitem__(self, z: int) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        img = torch.from_numpy(self._img[z].copy())
        lbl = torch.from_numpy(self._lbl[z].copy())
        sv  = torch.from_numpy(self._sv[z].copy()) if self._sv is not None else None
        return img, lbl, sv

    @property
    def n_slices(self) -> int:
        return self._img.shape[0]


class SliceDataset:
    """
    Loads a set of NIfTI volumes into RAM and exposes slice-level access.

    Usage:
        ds = SliceDataset(data_dir, scan_ids)
        img, lbl, sv = ds['001'][15]   # → three [H, W] tensors

    Normalization: per-volume zero-mean / unit-std (suitable for MR).
    """

    def __init__(
        self,
        data_dir: str,
        scan_ids: list[str],
        sv_prefix: str = 'MIDDLE',
        normalize: bool = True,
    ):
        self._scans: dict[str, _ScanView] = {}

        for sid in scan_ids:
            img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
            lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)

            sv_path = os.path.join(data_dir, f'superpix-{sv_prefix}_{sid}.nii.gz')
            sv = _read_nii(sv_path).astype(np.int32) if os.path.exists(sv_path) else None

            if normalize:
                img = (img - img.mean()) / (img.std() + 1e-8)

            self._scans[sid] = _ScanView(img, lbl, sv)

    def __getitem__(self, scan_id: str) -> _ScanView:
        return self._scans[scan_id]

    def __contains__(self, scan_id: str) -> bool:
        return scan_id in self._scans

    def scan_ids(self) -> list[str]:
        return list(self._scans.keys())


class EpisodeDataset(Dataset):
    """
    Samples few-shot segmentation episodes.

    Training   (use_gt=False): classmap_1.json,    mask = (sv == sv_id)
    Validation (use_gt=True):  gt_classmap_1.json, mask = (lbl == organ_idx)

    Each episode returns:
        support_imgs:  [K, H, W]  float32
        support_masks: [K, H, W]  float32 binary
        query_img:     [H, W]     float32
        query_mask:    [H, W]     float32 binary
        class_key:     str

    Support and query always come from different scans to prevent leakage.
    When n_shot > (available scans - 1), support is sampled with replacement.
    """

    def __init__(
        self,
        data_dir: str,
        scan_ids: list[str],
        n_shot: int = 1,
        n_episodes: int = 1000,
        use_gt: bool = False,
        augment: bool = False,
        label_names: Optional[list[str]] = None,
        sv_prefix: str = 'MIDDLE',
        min_px_key: str = '1',
    ):
        self.n_shot     = n_shot
        self.n_episodes = n_episodes
        self.use_gt     = use_gt
        self.transform  = get_train_transform() if augment else None

        self.slices = SliceDataset(data_dir, scan_ids, sv_prefix)

        # organ name -> label index, needed when use_gt=True
        self._label_idx: dict[str, int] = (
            {name: idx for idx, name in enumerate(label_names)}
            if label_names else {}
        )

        # load classmap and filter to this split's scan IDs
        cm_name = f'gt_classmap_{min_px_key}.json' if use_gt else f'classmap_{min_px_key}.json'
        with open(os.path.join(data_dir, cm_name)) as f:
            raw = json.load(f)

        # here filter classmap to selected id
        sid_set = set(scan_ids)
        self.classmap: dict[str, dict[str, list[int]]] = {}
        for cls_key, scan_dict in raw.items():
            filtered = {sid: zs for sid, zs in scan_dict.items() if sid in sid_set and zs}
            if len(filtered) >= 2:  # need >= 2 scans so support ≠ query
                self.classmap[cls_key] = filtered

        self.class_keys = list(self.classmap.keys())
        if not self.class_keys:
            raise ValueError(f'No valid classes in {cm_name} for the given scan_ids')

    def __len__(self) -> int:
        return self.n_episodes

    def __getitem__(self, _) -> dict:
        return self._sample_episode()

    def _sample_episode(self) -> dict:
        cls_key   = random.choice(self.class_keys) # select random sp class
        scan_dict = self.classmap[cls_key] # found dicts {scan: z slice} related to sp id
        all_scans = list(scan_dict.keys()) # all scans related to sp id as list

        query_scan   = random.choice(all_scans) # choice random query scan
        support_pool = [s for s in all_scans if s != query_scan]
        support_scans = random.choices(support_pool, k=self.n_shot)

        support_imgs, support_masks = [], []
        for sid in support_scans:
            z = random.choice(scan_dict[sid])
            img, lbl, sv = self.slices[sid][z]
            mask = self._make_mask(lbl, sv, cls_key).float()
            img, mask = self._apply_transform(img, mask)
            support_imgs.append(img)
            support_masks.append(mask)

        q_z = random.choice(scan_dict[query_scan])
        q_img, q_lbl, q_sv = self.slices[query_scan][q_z]
        q_mask = self._make_mask(q_lbl, q_sv, cls_key).float()
        q_img, q_mask = self._apply_transform(q_img, q_mask)

        return {
            'support_imgs':  torch.stack(support_imgs),
            'support_masks': torch.stack(support_masks),
            'query_img':     q_img,
            'query_mask':    q_mask,
            'class_key':     cls_key,
        }

    def _apply_transform(
        self,
        img: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.transform is None:
            return img, mask
        out = self.transform({'img': img, 'mask': mask})
        # EnsureChannelFirstd adds dim → squeeze back to [H, W]
        return out['img'].squeeze(0), out['mask'].squeeze(0)

    def _make_mask(
        self,
        lbl: torch.Tensor,
        sv: Optional[torch.Tensor],
        cls_key: str,
    ) -> torch.Tensor:
        if self.use_gt:
            idx = self._label_idx.get(cls_key, int(cls_key))
            return (lbl == idx).long()
        return (sv == int(cls_key)).long()