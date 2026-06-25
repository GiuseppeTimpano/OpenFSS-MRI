"""
Test-time translation: converts a query MRI volume from source → target domain.
Used as a hook in test.py before the segmenter receives the query image.
"""

import numpy as np
import torch

from cyclegan.models import UNetGenerator2D
from cyclegan.dataset import normalize_slice


def load_generator(
    ckpt_path: str,
    key: str = 'G_AB',
    device: str | torch.device = 'cpu',
) -> UNetGenerator2D:
    """Load a trained generator from a CycleGAN checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    G = UNetGenerator2D()
    G.load_state_dict(ckpt[key])
    G.to(device)
    G.eval()
    return G


def translate_volume(
    volume_npy: np.ndarray,
    G: UNetGenerator2D,
    device: torch.device,
) -> np.ndarray:
    """
    Translate a z-score-normalized MRI volume slice by slice via a CycleGAN generator.

    Normalization pipeline:
      Input  : z-score normalized (output of _load_scan in test.py)
      CycleGAN input  : per-slice percentile clip + min-max → [0, 1]
      CycleGAN output : tanh → [-1, 1] → rescaled [0, 1]
      Output : z-score normalized (matches segmenter training distribution)

    volume_npy : [D, H, W] float32
    Returns    : [D, H, W] float32
    """
    D = volume_npy.shape[0]
    translated = np.zeros_like(volume_npy)

    G.eval()
    with torch.no_grad():
        for z in range(D):
            sl_01 = normalize_slice(volume_npy[z])  # percentile + minmax → [0,1]
            inp   = torch.from_numpy(sl_01).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            out   = G(inp)                                                          # tanh [-1,1]
            translated[z] = (out.squeeze().cpu().numpy() + 1.0) / 2.0             # → [0,1]

    # Re-apply z-score so the segmenter receives the expected input distribution
    mu  = translated.mean()
    std = translated.std() + 1e-8
    return ((translated - mu) / std).astype(np.float32)
