"""
BiomedParse (v2) adapter for the foundation-model comparison (text-prompted, 3D).

BiomedParse v2 (Nature Methods + BoltzFormer arch, repo microsoft/BiomedParse) is a
text-prompted foundation segmenter: given a 3D volume + a free-text description of
the target organ it predicts a per-slice mask end-to-end. No box/point prompt, no
support set — the third paradigm alongside prompt-box (MedSAM2) and support-set
(UniverSeg/prototype), see HANDOFF.md.

NOT pip-installable (no setup.py/pyproject upstream): vendored as a pinned git
submodule (`third_party/BiomedParse`) and imported by adding its root to sys.path +
hydra config-dir compose, mirroring `third_party/BiomedParse/inference.py`'s own
CLI entrypoint (the "Inference 3D Examples" section of the upstream README).

Checkpoint: HuggingFace `microsoft/BiomedParse` (`biomedparse_v2.ckpt`), downloaded
lazily via `huggingface_hub.hf_hub_download` on first use (large, cached under the
default HF cache dir) unless a local path is passed explicitly.

Input contract (see third_party/BiomedParse/inference.py `main()` and utils.py):
  volume -> utils.process_input(vol, 512): pads to square (in-plane) + bicubic
  resize to 512x512, cast to int (the model does its own internal normalization,
  NOT the z-score/ImageNet normalization the other adapters use).
  text prompt -> one descriptive sentence for the target organ; we always call with
  exactly ONE class (one organ at a time, like the other adapters' per-label loop),
  so no "[SEP]" multi-prompt joining / merge_multiclass_masks argmax is needed — a
  simple sigmoid>0.5 threshold on the single class channel is equivalent (verified
  against inference.py's own merge_multiclass_masks: for N=1 it argmaxes against a
  constant 0.5 background channel, which is the same threshold).

Contamination note: v2 pretraining (CVPR-BiomedSegFM, junma/CVPR-BiomedSegFM)
includes AMOS and the TotalSegmentator MRI split — see HANDOFF.md contamination
matrix. NOT clean on CHAOS/AMOS/TS-MRI; only usable on datasets released after that
training corpus was frozen (CirrMRI, per the project's existing timeline argument).
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
