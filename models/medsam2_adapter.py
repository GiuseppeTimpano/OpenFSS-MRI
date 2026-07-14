"""
MedSAM2 adapter -- promptable (box + SAM2 memory-attention propagation across
slices), NOT support-set few-shot like models/fewshot.py; wrapped at volume level.

Predictor API mirrors MedSAM2's medsam2_infer_3D_CT.py: init_state ->
add_new_points_or_box (one box on one slice) -> propagate_in_video (both directions).

Normalization is a MedSAM2 requirement, not a bug: uint8 [0,255], resized to 512,
ImageNet standardization -- NOT the per-volume z-score of
scripts/prototype/test.py's _load_scan.

`sam2`/MedSAM2 imported lazily so this module loads even where they're absent.
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


def mask_to_box(mask: np.ndarray, margin: int = 0):
    """mask: [H,W] bool. Tight bbox (x0,y0,x1,y1) + margin, or None if mask empty.
    Used for cascaded point->box iterative refinement (MedSAM2Segmenter.segment_volume_points)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    H, W = mask.shape
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(W - 1, int(xs.max()) + margin)
    y1 = min(H - 1, int(ys.max()) + margin)
    return (float(x0), float(y0), float(x1), float(y1))


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
    def embed_frame(self, frame_u8: np.ndarray) -> torch.Tensor:
        """frame_u8: [H,W] uint8 -> [C,h,w] SAM2 image-encoder feature (most-semantic
        FPN level), for PerSAM-style prototype/similarity matching (models/support_prompt.py).
        Independent of any video/inference_state -- just runs the image encoder."""
        img = self._preprocess(frame_u8[None])  # [1,3,IMG_SIZE,IMG_SIZE]
        autocast = (torch.autocast(self.device_type, dtype=torch.bfloat16)
                    if self.device_type == "cuda"
                    else torch.autocast(self.device_type, enabled=False))
        with autocast:
            out = self.predictor.forward_image(img)
        return out["backbone_fpn"][-1][0]

    @torch.inference_mode()
    def segment_volume(self, vol_u8: np.ndarray,
                       boxes: dict[int, np.ndarray],
                       refine_iters: int = 0,
                       neg_points: dict[int, list] | None = None) -> np.ndarray:
        """
        vol_u8 : [Z,H,W] uint8 [0,255] (already cropped to the propagation range).
        boxes  : {frame_idx -> [x0,y0,x1,y1]} in ORIGINAL (H,W) coords; one or more
                 prompted slices. Propagation conditions on all prompted frames.
        refine_iters : cascaded box refinement on each prompted frame BEFORE
                       propagation: derive a tighter box from the box-prompt mask
                       preview (single-frame decode, no propagation), re-prompt with
                       that box, repeat until the box stabilizes or refine_iters is
                       hit. 0 = no refinement (single box-prompt pass, previous
                       behavior). Same pattern as segment_volume_points' refine_iters,
                       ported to the box-prompt path.
        neg_points : optional {frame_idx -> [(x,y), ...]} negative clicks added on top
                     of the box on the same frame (box+neg-points hybrid, models/
                     support_prompt.py multiclass_boxes(neg_points=True)) -- meant to
                     push SAM2 off neighboring muscle that an elongated box also covers.
                     None/missing frame = box-only, previous behavior.
        returns: [Z,H,W] uint8 binary mask.
        """
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        if not boxes:
            return seg
        neg_points = neg_points or {}

        img_resized = self._preprocess(vol_u8)
        autocast = (torch.autocast(self.device_type, dtype=torch.bfloat16)
                    if self.device_type == "cuda"
                    else torch.autocast(self.device_type, enabled=False))
        with autocast:
            state = self.predictor.init_state(img_resized, H, W)
            for fidx, box in sorted(boxes.items()):
                cur_box = tuple(float(v) for v in box)
                pts = neg_points.get(fidx)
                pts_arr = np.asarray(pts, dtype=np.float32) if pts else None
                lbl_arr = np.zeros(len(pts), dtype=np.int32) if pts else None
                _, _, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=int(fidx), obj_id=1,
                    box=np.asarray(cur_box, dtype=np.float32),
                    points=pts_arr, labels=lbl_arr)

                for _ in range(refine_iters):
                    mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
                    new_box = mask_to_box(mask)
                    if new_box is None or new_box == cur_box:
                        break
                    cur_box = new_box
                    _, _, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=int(fidx), obj_id=1,
                        box=np.asarray(cur_box, dtype=np.float32),
                        points=pts_arr, labels=lbl_arr)

            for reverse in (False, True):
                for fidx, _oids, logits in self.predictor.propagate_in_video(
                        state, reverse=reverse):
                    seg[fidx][(logits[0] > 0.0).cpu().numpy()[0]] = 1
        return seg

    @torch.inference_mode()
    def segment_volume_points(self, vol_u8: np.ndarray,
                              points: dict[int, tuple],
                              refine_iters: int = 0) -> np.ndarray:
        """
        Point-prompted variant of segment_volume, for prompt_mode=support (see
        models/support_prompt.py) -- does NOT touch/replace segment_volume (box-oracle
        path, prompt_mode in {perslice,key}), added alongside it.

        vol_u8       : [Z,H,W] uint8 [0,255].
        points       : {frame_idx -> (pos_xy, neg_xy)}, each xy in ORIGINAL (H,W) coords,
                       label 1=positive/0=negative.
        refine_iters : cascaded PerSAM-style refinement on each prompted (key) frame BEFORE
                       propagation: derive a tight box from the point-prompt mask, re-prompt
                       with that box (SAM2 feeds the previous mask logits into the decoder
                       automatically), repeat until the box stabilizes or refine_iters is hit.
                       0 = no refinement (single point-prompt pass), matching the plan's
                       original scope.
        returns: [Z,H,W] uint8 binary mask.
        """
        Z, H, W = vol_u8.shape
        seg = np.zeros((Z, H, W), dtype=np.uint8)
        if not points:
            return seg

        img_resized = self._preprocess(vol_u8)
        autocast = (torch.autocast(self.device_type, dtype=torch.bfloat16)
                    if self.device_type == "cuda"
                    else torch.autocast(self.device_type, enabled=False))
        with autocast:
            state = self.predictor.init_state(img_resized, H, W)
            for fidx, (pos_xy, neg_xy) in sorted(points.items()):
                pts = np.asarray([pos_xy, neg_xy], dtype=np.float32)
                labels = np.asarray([1, 0], dtype=np.int32)
                _, _, mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=int(fidx), obj_id=1,
                    points=pts, labels=labels)

                box = None
                for _ in range(refine_iters):
                    mask = (mask_logits[0, 0] > 0.0).cpu().numpy()
                    new_box = mask_to_box(mask)
                    if new_box is None or new_box == box:
                        break
                    box = new_box
                    _, _, mask_logits = self.predictor.add_new_points_or_box(
                        inference_state=state, frame_idx=int(fidx), obj_id=1,
                        box=np.asarray(box, dtype=np.float32))

            for reverse in (False, True):
                for fidx, _oids, logits in self.predictor.propagate_in_video(
                        state, reverse=reverse):
                    seg[fidx][(logits[0] > 0.0).cpu().numpy()[0]] = 1
        return seg
