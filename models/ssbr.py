"""
Self-Supervised Body Part Regression (SSBR), Yan et al. 2018.

A small CNN f(slice) -> scalar that encodes the body-axis position of an axial
slice. Trained self-supervised from within-volume equidistant ordered slices:
no labels required.

Losses (per sampled equidistant sequence of M slices, scores s_0..s_{M-1}):
  - order loss:    -mean log sigmoid(s_{j+1} - s_j)   -> scores increase along axis
  - equidist loss:  mean ( d_j - mean(d) )^2 , d_j=s_{j+1}-s_j  -> linear in position

Use at FSS test time: score query + support slices, match a query slice to the
support slice with the nearest body-part score. No query label needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SSBRNet(nn.Module):
    """Small strided-conv regressor: [N,1,RES,RES] -> [N] scalar score."""

    def __init__(self):
        super().__init__()

        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, 2, 1),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        self.body = nn.Sequential(
            blk(1, 16), blk(16, 32), blk(32, 64), blk(64, 128), blk(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x).flatten(1)
        return self.head(h).squeeze(-1)


def ssbr_order_equidist_loss(scores: torch.Tensor, alpha: float = 1.0):
    """SSBR loss for one ordered equidistant sequence of scores [M]."""
    d = scores[1:] - scores[:-1]
    loss_order = -F.logsigmoid(d).mean()
    loss_equidist = ((d - d.mean()) ** 2).mean()
    return loss_order + alpha * loss_equidist, loss_order, loss_equidist


@torch.no_grad()
def score_volume(net: SSBRNet, vol: np.ndarray, res: int,
                 device: torch.device, batch: int = 64) -> np.ndarray:
    """
    Score every axial slice of a normalized volume -> [Z] body-part scores.
    `vol` is [Z,H,W], already per-volume normalized (same as the FSS pipeline).
    """
    net.eval()
    x = torch.from_numpy(vol.astype(np.float32)).unsqueeze(1)        # [Z,1,H,W]
    x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
    out = []
    for i in range(0, x.shape[0], batch):
        out.append(net(x[i:i + batch].to(device)).cpu())
    return torch.cat(out).numpy()                                   # [Z]


def load_ssbr(ckpt_path: str, device: torch.device) -> tuple[SSBRNet, int]:
    """Load a trained SSBR checkpoint. Returns (net, res)."""
    ck = torch.load(ckpt_path, map_location='cpu')
    net = SSBRNet().to(device)
    net.load_state_dict(ck['state_dict'])
    net.eval()
    res = int(ck.get('res', 128))
    return net, res
