"""
BiomedParse (v2) adapter -- text-prompted, 3D, no box/support set (third paradigm
alongside MedSAM2 box-prompt and UniverSeg/prototype support-set, see HANDOFF.md).

Vendored as git submodule (third_party/BiomedParse, not pip-installable); imported
via sys.path + hydra config-dir compose. Checkpoint downloaded lazily from HF
(microsoft/BiomedParse) unless a local path is passed.

Input: utils.process_input(vol, 512) (pad+resize, model's own normalization, NOT
z-score/ImageNet). We call with exactly one class per volume, so plain
sigmoid>0.5 is equivalent to upstream's merge_multiclass_masks for N=1.

CONTAMINATION: v2 pretraining includes AMOS + TotalSegmentator-MRI (HANDOFF.md
matrix). Not clean on CHAOS/AMOS/TS-MRI; only CirrMRI is usable here.
"""
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'third_party', 'BiomedParse'))

IMG_SIZE = 512

# MRI-phrased prompts, mirroring the CT phrasing style seen in BiomedParse's own
# CVPR-BiomedSegFM training data (examples/imgs/*.npz text_prompts), "CT" -> "MRI".
# Kidney isn't listed under BiomedParse's documented MRI task list (README
# "Supported Tasks") but IS one of the model's generic organ classes (see
# third_party/BiomedParse/src/model/biomedparse_3D.py MaskFormerHead.classes) —
# treat RK/LK as best-effort, flag if Dice looks degenerate.
PROMPT_TEMPLATES = {
    'LIVER':  'Visualization of the liver in abdominal MRI imaging',
    'RK':     'Presence of the right kidney detected in abdominal MRI images',
    'LK':     'Abdominal MRI showing the left kidney',
    'SPLEEN': 'MRI imaging of the spleen within the abdomen',
}


def volume_to_uint8(vol: np.ndarray, p_low: float = 0.5, p_high: float = 99.5) -> np.ndarray:
    """Raw float MRI volume -> uint8 [0,255], per-volume percentile clip + min-max
    (same windowing convention as models.medsam2_adapter.volume_to_uint8)."""
    lo, hi = np.percentile(vol, [p_low, p_high])
    v = np.clip(vol, lo, hi)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8) * 255.0
    return v.astype(np.uint8)


class BiomedParseSegmenter:
    """Volume-level BiomedParse wrapper: one text prompt -> per-slice mask for the
    whole volume in one forward pass (no support set, no box/point prompt)."""

    def __init__(self, checkpoint: str | None = None, device: str = "cuda"):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        import hydra
        from hydra import compose
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        hydra.initialize_config_dir(config_dir=os.path.join(_REPO_ROOT, 'configs', 'model'),
                                    version_base=None)
        cfg = compose(config_name='biomedparse_3D')
        self.model = hydra.utils.instantiate(cfg, _convert_='object')

        ckpt = checkpoint
        if ckpt is None:
            from huggingface_hub import hf_hub_download
            ckpt = hf_hub_download(repo_id='microsoft/BiomedParse',
                                   filename='biomedparse_v2.ckpt')
        self.model.load_pretrained(ckpt)
        self.model.to(device).eval()
        self.device = device

    @torch.inference_mode()
    def segment_volume(self, vol_u8: np.ndarray, text_prompt: str,
                       slice_batch_size: int = 4) -> np.ndarray:
        """
        vol_u8      : [Z,H,W] uint8 [0,255] (already cropped to the FG depth range).
        text_prompt : one descriptive sentence for the target organ, e.g.
                      PROMPT_TEMPLATES['LIVER'].
        returns     : [Z,H,W] uint8 binary mask at the ORIGINAL (H,W) resolution.
        """
        from utils import process_input, process_output        # BiomedParse repo root
        from inference import postprocess                      # BiomedParse repo root

        imgs, pad_width, padded_size, valid_axis = process_input(vol_u8, IMG_SIZE)
        imgs = imgs.to(self.device).int()

        input_tensor = {"image": imgs.unsqueeze(0), "text": [text_prompt]}
        output = self.model(input_tensor, mode="eval", slice_batch_size=slice_batch_size)

        mask_preds = output["predictions"]["pred_gmasks"]
        mask_preds = F.interpolate(mask_preds, size=(IMG_SIZE, IMG_SIZE),
                                   mode="bicubic", align_corners=False, antialias=True)
        mask_preds = postprocess(mask_preds, output["predictions"]["object_existence"],
                                 do_nms=False)              # single class: no NMS needed
        mask = (mask_preds[0] > 0.5).int()                  # [D,H,W] in padded/resized space

        mask = process_output(mask, pad_width, padded_size, valid_axis)
        return mask.astype(np.uint8)
