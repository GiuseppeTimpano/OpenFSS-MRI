"""
UniverSeg adapter for the foundation-model comparison (in-context FSS, deployable —
no oracle box, just a support set: same information budget as the prototype baseline).

UniverSeg (ICCV 2023, arXiv 2304.06131, repo JJGO/UniverSeg) is a fixed pretrained
model with NO per-task training or fine-tuning: it takes a target slice + a support
set (images + binary labels) and predicts the target mask directly via cross-attention
between target and support, jointly over the whole support set at once (not a
metric/prototype match like ALPNet/QNet). Native input size is fixed at 128x128,
1-channel, pixel values min-max normalized to [0,1] (see repo README + example_data/
oasis.py: percentile/`nib` scan -> [0,1] float -> PIL bilinear resize to 128x128;
labels resized nearest).

Output is raw logits (no sigmoid applied internally, see universeg/model.py
`out_activation=None`) -> apply sigmoid then threshold at 0.5.

`universeg` is vendored as a pinned git submodule (`third_party/UniverSeg`, same pattern
as `third_party/MedSAM2`) and installed editable (`pip install -e third_party/UniverSeg`).
It downloads/caches its pretrained weights via `torch.hub` on first use. Imported eagerly
here since, unlike `sam2` (models/medsam2_adapter.py), it has no heavy optional deps.
"""
import numpy as np
import torch
import torch.nn.functional as F

IMG_SIZE = 128


def volume_to_unit_float(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Raw float MRI volume -> float32 [0,1], per-volume percentile clip + min-max.
    Same windowing as models.medsam2_adapter.volume_to_uint8, kept in [0,1] instead of
    [0,255] since UniverSeg expects [0,1] float input (no ImageNet standardization)."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    return ((v - v.min()) / (v.max() - v.min() + 1e-8)).astype(np.float32)


class UniverSegSegmenter:
    """Joint-context UniverSeg wrapper: a fixed support set (images + binary masks) is
    used to predict every target slice in one forward pass per slice (batched)."""

    def __init__(self, device: str = "cuda", pretrained: bool = True):
        from universeg import universeg  # small pip pkg, weights via torch.hub cache
        self.model = universeg(pretrained=pretrained).to(device).eval()
        self.device = device

    def _resize(self, x: torch.Tensor, size: tuple[int, int], mode: str) -> torch.Tensor:
        kwargs = {} if mode == "nearest" else {"align_corners": False}
        return F.interpolate(x, size=size, mode=mode, **kwargs)

    @torch.inference_mode()
    def segment_volume(self, q_vol01: np.ndarray, supp_imgs01: np.ndarray,
                       supp_masks: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """
        q_vol01     : [Z,H,W] float32 [0,1] query volume (already cropped to FG range).
        supp_imgs01 : [S,H,W] float32 [0,1] support images (same support set for every
                      query slice — this is UniverSeg's native in-context usage).
        supp_masks  : [S,H,W] {0,1} support binary labels for the target organ.
        returns     : [Z,H,W] uint8 binary mask at the ORIGINAL (H,W) resolution.
        """
        Z, H, W = q_vol01.shape
        S = supp_imgs01.shape[0]
        dev = self.device

        supp_img_t  = torch.from_numpy(supp_imgs01).to(dev).float().unsqueeze(1)   # [S,1,H,W]
        supp_mask_t = torch.from_numpy(supp_masks).to(dev).float().unsqueeze(1)    # [S,1,H,W]
        supp_img_t  = self._resize(supp_img_t,  (IMG_SIZE, IMG_SIZE), "bilinear")
        supp_mask_t = self._resize(supp_mask_t, (IMG_SIZE, IMG_SIZE), "nearest")

        target_t = torch.from_numpy(q_vol01).to(dev).float().unsqueeze(1)          # [Z,1,H,W]
        target_t = self._resize(target_t, (IMG_SIZE, IMG_SIZE), "bilinear")

        probs = torch.empty(Z, 1, IMG_SIZE, IMG_SIZE, device=dev)
        for i in range(0, Z, batch_size):
            tb = target_t[i:i + batch_size]
            b = tb.shape[0]
            sup_img_b  = supp_img_t.unsqueeze(0).expand(b, S, 1, IMG_SIZE, IMG_SIZE)
            sup_mask_b = supp_mask_t.unsqueeze(0).expand(b, S, 1, IMG_SIZE, IMG_SIZE)
            logits = self.model(tb, sup_img_b, sup_mask_b)
            probs[i:i + b] = torch.sigmoid(logits)

        probs_full = self._resize(probs, (H, W), "bilinear")
        return (probs_full[:, 0] > 0.5).cpu().numpy().astype(np.uint8)
