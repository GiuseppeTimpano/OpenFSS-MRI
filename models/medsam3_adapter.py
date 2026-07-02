"""
MedSAM3 adapter -- text-prompted, 3D via slice-wise loop, ZERO-SHOT only (no
further fine-tuning here). Loads MedSAM3-v1's published LoRA weights
(huggingface.co/lal-Joey/MedSAM3_v1) on top of facebook/sam3. Same paradigm slot
as biomedparse_adapter.py (text, no box, no support set) -- see HANDOFF.md.

Vendored as git submodule (third_party/MedSAM3, not pip-installable); this
adapter inlines third_party/MedSAM3/infer_sam.py's SAM3LoRAInference.predict()
logic to operate on in-memory PIL slices. Upstream has no volumetric/NIfTI
handling -- the slice loop is this project's addition.

CONTAMINATION: MedSAM3-v1's LoRA training set is unpublished/proprietary (arXiv
2511.19046 §4.1). No CHAOS/AMOS/CirrMRI overlap found by grep, but this is an
open/unverifiable risk, not confirmed-clean -- report as "leakage unknown", not
"clean".

LICENSE: upstream has no LICENSE file; base facebook/sam3 weights (gated HF
download) carry Meta's own license. Fine for local experimentation; check before
redistributing weights or publishing results externally.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image as PILImage
from torchvision.ops import nms

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'MedSAM3'))

# MRI-phrased prompts, same style/rationale as biomedparse_adapter.PROMPT_TEMPLATES.
PROMPT_TEMPLATES = {
    'LIVER':  'liver',
    'RK':     'right kidney',
    'LK':     'left kidney',
    'SPLEEN': 'spleen',
}


def volume_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Raw float MRI volume -> uint8 [0,255], per-volume percentile clip + min-max
    (same windowing convention as models.medsam2_adapter.volume_to_uint8)."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255.0
    return v.astype(np.uint8)


def build_medsam3_lora_model(config_path: str | None, weights_path: str | None,
                              device: torch.device, use_lora: bool = True):
    """Build SAM3 + apply the published MedSAM3-v1 LoRA weights (zero-shot,
    no fine-tuning by this project). Shared by MedSAM3Segmenter (single
    forward pass) and models.medsam3_agent_adapter.MedSAM3AgentSegmenter
    (same weights, driven through the multi-round agent loop instead) so the
    two never drift out of sync on how the LoRA model is built.

    use_lora=False skips the MedSAM3-v1 LoRA entirely and returns plain
    pretrained facebook/sam3 (base HF weights, load_from_HF=True below) --
    baseline to tell apart "grounding head can't localize small organs"
    (architectural) from "LoRA weights specifically hurt kidney/spleen"
    (training-data/overfit) when the LoRA'd model's RK/LK/SPLEEN dice is bad
    even on scans it may have trained on (see results/medsam3/*/scores.csv)."""
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        device=device.type,
        compile=False,
        load_from_HF=True,
        bpe_path=os.path.join(_REPO_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        eval_mode=True,
    )

    if use_lora:
        from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

        if config_path is None:
            config_path = os.path.join(_REPO_ROOT, 'configs', 'full_lora_config.yaml')
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        if weights_path is None:
            # Auto-download the paper authors' published LoRA weights (zero-shot,
            # no fine-tuning by this project) from HuggingFace.
            from huggingface_hub import hf_hub_download
            weights_path = hf_hub_download(
                repo_id='lal-Joey/MedSAM3_v1', filename='best_lora_weights.pt')

        lora_cfg = config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        model = apply_lora_to_model(model, lora_config)
        load_lora_weights(model, weights_path)

    model.to(device)
    model.eval()
    return model


class MedSAM3Segmenter:
    """Volume-level MedSAM3 wrapper: one text prompt -> per-slice mask, looping
    the upstream 2D-only LoRA model slice-by-slice (no native volume forward
    pass, unlike BiomedParse)."""

    def __init__(
        self,
        config_path: str | None = None,
        weights_path: str | None = None,
        resolution: int = 1008,
        detection_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        device: str = "cuda",
        use_lora: bool = False,
    ):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)

        from sam3.train.data.sam3_image_dataset import (
            Datapoint, Image as SAMImage, FindQueryLoaded, InferenceMetadata,
        )
        from sam3.train.data.collator import collate_fn_api
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI,
        )

        self._SAMImage = SAMImage
        self._FindQueryLoaded = FindQueryLoaded
        self._InferenceMetadata = InferenceMetadata
        self._Datapoint = Datapoint
        self._collate_fn_api = collate_fn_api
        self._copy_data_to_device = copy_data_to_device

        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.model = build_medsam3_lora_model(config_path, weights_path, self.device, use_lora=use_lora)

        self.transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=resolution, max_size=resolution,
                             square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    @torch.no_grad()
    def _predict_slice(self, pil_image: PILImage.Image, text_prompt: str) -> np.ndarray:
        """Single 2D slice -> binary mask [H,W] uint8, original resolution.
        Only the single highest-score detection is kept (one-organ-per-slice
        assumption, matching the other adapters' single-class-per-call setup)."""
        w, h = pil_image.size
        sam_image = self._SAMImage(data=pil_image, objects=[], size=[h, w])
        query = self._FindQueryLoaded(
            query_text=text_prompt, image_id=0, object_ids_output=[],
            is_exhaustive=True, query_processing_order=0,
            inference_metadata=self._InferenceMetadata(
                coco_image_id=0, original_image_id=0, original_category_id=1,
                original_size=[w, h], object_id=0, frame_index=0,
            ),
        )
        datapoint = self._Datapoint(find_queries=[query], images=[sam_image])
        datapoint = self.transform(datapoint)
        batch = self._collate_fn_api([datapoint], dict_key="input")["input"]
        batch = self._copy_data_to_device(batch, self.device, non_blocking=True)

        outputs = self.model(batch)
        last_output = outputs[-1]
        pred_logits = last_output['pred_logits']
        pred_boxes = last_output['pred_boxes']
        pred_masks = last_output.get('pred_masks', None)

        scores = pred_logits.sigmoid()[0, :, :].max(dim=-1)[0]
        keep = scores > self.detection_threshold
        if keep.sum().item() == 0 or pred_masks is None:
            return np.zeros((h, w), dtype=np.uint8)

        boxes_cxcywh = pred_boxes[0, keep]
        kept_scores = scores[keep]
        cx, cy, bw, bh = boxes_cxcywh.unbind(-1)
        x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h
        x2, y2 = (cx + bw / 2) * w, (cy + bh / 2) * h
        boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        keep_nms = nms(boxes_xyxy, kept_scores, self.nms_iou_threshold)
        kept_scores = kept_scores[keep_nms]
        masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5

        best = kept_scores.argmax().item()
        mask_small = masks_small[best:best + 1].unsqueeze(0).float()
        mask = F.interpolate(mask_small, size=(h, w), mode='bilinear',
                              align_corners=False).squeeze() > 0.5
        return mask.cpu().numpy().astype(np.uint8)

    def segment_volume(self, vol_u8: np.ndarray, text_prompt: str) -> np.ndarray:
        """
        vol_u8      : [Z,H,W] uint8 [0,255] (already cropped to the FG depth range).
        text_prompt : short noun phrase, e.g. PROMPT_TEMPLATES['LIVER'].
        returns     : [Z,H,W] uint8 binary mask at the ORIGINAL (H,W) resolution.
        """
        z, h, w = vol_u8.shape
        out = np.zeros((z, h, w), dtype=np.uint8)
        for i in range(z):
            pil_slice = PILImage.fromarray(vol_u8[i]).convert("RGB")
            out[i] = self._predict_slice(pil_slice, text_prompt)
        return out
