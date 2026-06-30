"""
MedSAM2 adapter for the foundation-model comparison.

MedSAM2 (SAM2.1 fine-tuned, arXiv 2504.03600, repo bowang-lab/MedSAM2) is NOT a
few-shot-from-support model like the prototype baseline. It is *promptable*: it
takes a box on one slice of the query volume and propagates the mask to the other
slices via SAM2 memory attention. So it is wrapped at the volume level (not the
per-slice forward(support, query) contract of models/fewshot.py).

The exact predictor API mirrors MedSAM2's medsam2_infer_3D_CT.py:
  predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
  state     = predictor.init_state(img_resized, H, W)   # img_resized: [Z,3,512,512]
  predictor.add_new_points_or_box(inference_state=state, frame_idx=z, obj_id=1,
                                  box=np.array([x0,y0,x1,y1]))
  for fidx, oids, logits in predictor.propagate_in_video(state[, reverse=True]):
      mask = (logits[0] > 0.0)[0]

Normalization differs from the baseline on purpose: MedSAM2 wants uint8 [0,255]
volumes resized to 512 + ImageNet standardization (NOT the per-volume z-score of
test._load_scan). This is a model requirement, not a bug.

`sam2` / MedSAM2 live in their own repo and are imported lazily so this module
can be imported even where they are not installed.
"""
import numpy as np
import torch
from PIL import Image

IMG_SIZE = 512
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def resize_grayscale_to_rgb_and_resize(array: np.ndarray, image_size: int) -> np.ndarray:
    """[Z,H,W] uint8 -> [Z,3,image_size,image_size] float (0..255). Mirrors MedSAM2."""
    d = array.shape[0]
    out = np.zeros((d, 3, image_size, image_size), dtype=np.float32)
    for i in range(d):
        img = Image.fromarray(array[i].astype(np.uint8)).convert("RGB")
        img = img.resize((image_size, image_size))
        out[i] = np.array(img).transpose(2, 0, 1)
    return out


def volume_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Raw float MRI volume -> uint8 [0,255], per-volume percentile clip + min-max.
    (MRI has no DICOM window like the MedSAM2 CT script, so we robust-window here.)"""
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255.0
    return v.astype(np.uint8)


class MedSAM2Segmenter:
    """Volume-level MedSAM2 wrapper: box prompt(s) on chosen slice(s) + bidirectional
    propagation across the volume."""

    def __init__(self, checkpoint: str, model_cfg: str, device: str = "cuda"):
        from sam2.build_sam import build_sam2_video_predictor_npz  # lazy: MedSAM2 repo
        self.predictor = build_sam2_video_predictor_npz(model_cfg, checkpoint)
        self.device = device
        self.device_type = "cuda" if str(device).startswith("cuda") else "cpu"

    def _preprocess(self, vol_u8: np.ndarray) -> torch.Tensor:
        arr = resize_grayscale_to_rgb_and_resize(vol_u8, IMG_SIZE) / 255.0
        t = torch.from_numpy(arr).to(self.device).float()
        mean = torch.tensor(_IMAGENET_MEAN, device=self.device)[:, None, None]
        std  = torch.tensor(_IMAGENET_STD,  device=self.device)[:, None, None]
        return (t - mean) / std

    @torch.inference_mode()
    def segment_volume(self, vol_u8: np.ndarray,
                       boxes: dict[int, np.ndarray]) -> np.ndarray:
        """
        vol_u8 : [Z,H,W] uint8 [0,255] (already cropped to the propagation range).
        boxes  : {frame_idx -> [x0,y0,x1,y1]} in ORIGINAL (H,W) coords; one or more
                 prompted slices. Propagation conditions on all prompted frames.
        returns: [Z,H,W] uint8 binary mask.
        """
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        if not boxes:
            return seg

        img_resized = self._preprocess(vol_u8)
        autocast = (torch.autocast(self.device_type, dtype=torch.bfloat16)
                    if self.device_type == "cuda"
                    else torch.autocast(self.device_type, enabled=False))
        with autocast:
            state = self.predictor.init_state(img_resized, H, W)
            for fidx, box in sorted(boxes.items()):
                self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=int(fidx), obj_id=1,
                    box=np.asarray(box, dtype=np.float32))

            for reverse in (False, True):
                for fidx, _oids, logits in self.predictor.propagate_in_video(
                        state, reverse=reverse):
                    seg[fidx][(logits[0] > 0.0).cpu().numpy()[0]] = 1
        return seg
