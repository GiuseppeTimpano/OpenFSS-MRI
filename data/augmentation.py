"""
Augmentation built with monai but matching the original Q-Net / SSL-ALPNet protocol.

Original params (sabs config, identical in both repos):
  flip   : OFF (medical data has fixed orientation)
  affine : rotate +/-5 deg, shift +/-5 px, shear +/-5 deg, scale (0.9, 1.2)
  elastic: alpha 10, sigma 5  (small, smooth deformation)
  gamma  : range (0.5, 1.5)

Protocol (see datasets.py in both originals):
  - geom (affine + elastic) is applied to EITHER the support set OR the query (50/50)
  - gamma is applied to EITHER the support set OR the query (50/50)
  so gating is done by the caller; here prob=1.0 (transform always fires when called).
"""

import math

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandAffined,
    Rand2DElasticd,
    RandAdjustContrastd,
)

_DEG = math.pi / 180.0


def get_geom_transform() -> Compose:
    # affine + elastic on img AND mask together (same geometry).
    # mask uses nearest so labels stay binary.
    return Compose([
        # [H, W] -> [1, H, W] so spatial transforms can work
        EnsureChannelFirstd(keys=['img', 'mask'], channel_dim='no_channel'),
        RandAffined(
            keys=['img', 'mask'],
            mode=['bilinear', 'nearest'],
            prob=1.0,
            rotate_range=(5 * _DEG,),                  # +/-5 deg
            translate_range=(5, 5),                    # +/-5 px
            shear_range=(5 * _DEG, 5 * _DEG),          # +/-5 deg
            scale_range=((-0.1, 0.2), (-0.1, 0.2)),    # factor 0.9-1.2
            padding_mode='zeros',
        ),
        # monai elastic: spacing = control-grid step, magnitude = displacement.
        # small values approximate the original alpha=10 / sigma=5 deformation.
        Rand2DElasticd(
            keys=['img', 'mask'],
            mode=['bilinear', 'nearest'],
            prob=1.0,
            spacing=(20, 20),
            magnitude_range=(1, 3),
            padding_mode='zeros',
        ),
    ])


def get_intensity_transform() -> Compose:
    # gamma on img only (intensity, no geometry change -> mask untouched)
    return Compose([
        EnsureChannelFirstd(keys=['img'], channel_dim='no_channel'),
        RandAdjustContrastd(keys=['img'], prob=1.0, gamma=(0.5, 1.5)),
    ])
