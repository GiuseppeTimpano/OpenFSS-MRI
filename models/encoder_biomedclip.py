"""
BiomedCLIP encoder for the prototypical FSS heads (ALPNet / Q-Net).

Same role as encoder_dino.py: keep the prototypical pipeline fixed and swap only
the encoder. BiomedCLIP is a ViT-B/16 image encoder pretrained with CLIP on
PMC-15M (medical image-text pairs).

Differences from the DINO path:
  - loaded via open_clip (not torch.hub)
  - CLIP normalization stats (not ImageNet)
  - trained at 224x224 -> input is resized to 224 to avoid pos-embed mismatch

BiomedCLIPBackbone exposes the same API as DINOv3Backbone (vit path), so it plugs
into FoundationALPNetEncoder / FoundationQNetEncoder unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_dino import FoundationALPNetEncoder, FoundationQNetEncoder
from .lora import apply_lora


# CLIP normalization stats (BiomedCLIP uses the same as OpenAI CLIP).
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

_HF_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
_INPUT_SIZE = 224   # BiomedCLIP ViT-B/16 was trained at this resolution


def _load_biomedclip_trunk(model_name=_HF_NAME):
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    visual = model.visual
    return getattr(visual, 'trunk', visual)   # timm VisionTransformer


class BiomedCLIPBackbone(nn.Module):
    """ViT-B/16 BiomedCLIP image encoder with the DINOv3Backbone (vit) API.

    Provides:
      - extract_levels(x, indices) -> list of [B, C, h, w] (block indices)
      - global_vector(x)           -> [B, C] (GAP of the last level)
      - self.level_dims / n_levels / global_dim
    """

    arch = 'vit'

    def __init__(self, model_name=_HF_NAME, freeze=True, normalize=True,
                 lora_rank=0):
        super().__init__()
        self.normalize = normalize
        self.trunk = _load_biomedclip_trunk(model_name)

        if freeze:
            self.trunk.eval()
            for p in self.trunk.parameters():
                p.requires_grad = False
        self._frozen = freeze

        # LoRA (regime B): base frozen, PEFT adds trainable adapters -> run under grad.
        if lora_rank > 0:
            apply_lora(self.trunk, 'vit', rank=lora_rank)
            self._frozen = False

        self.register_buffer('_mean', torch.tensor(_CLIP_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(_CLIP_STD).view(1, 3, 1, 1))

        self.patch_size = 16
        self.level_dims, self.global_dim, self.n_levels = self._probe_dims()

    def _prep(self, x):
        # resize to the training resolution, then CLIP-normalize
        if x.shape[-2:] != (_INPUT_SIZE, _INPUT_SIZE):
            x = F.interpolate(x, size=(_INPUT_SIZE, _INPUT_SIZE),
                              mode='bilinear', align_corners=False)
        if self.normalize:
            x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        return x

    def _levels(self, x, indices):
        # timm ViT: get_intermediate_layers with reshape=True -> [B, C, h, w]
        feats = self.trunk.get_intermediate_layers(x, n=list(indices), reshape=True)
        return list(feats)

    def extract_levels(self, x, indices):
        x = self._prep(x)
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            return self._levels(x, list(indices))

    def global_vector(self, x):
        x = self._prep(x)
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            lvl = self._levels(x, [self.n_levels - 1])[0]
            return lvl.mean(dim=(2, 3))

    @torch.no_grad()
    def _probe_dims(self):
        was_training = self.trunk.training
        self.trunk.eval()
        dummy = torch.zeros(1, 3, _INPUT_SIZE, _INPUT_SIZE)
        if self.normalize:
            dummy = (dummy - self._mean) / self._std
        depth = len(getattr(self.trunk, 'blocks', range(12)))
        feats = self._levels(dummy, list(range(depth)))
        level_dims = [f.shape[1] for f in feats]
        if was_training:
            self.trunk.train()
        return level_dims, level_dims[-1], depth


# convenience builders (mirror the DINOv3 one-liners)
def biomedclip_alpnet_encoder(freeze=True, normalize=True, feat_index=None,
                              out_ch=256, lora_rank=0):
    bb = BiomedCLIPBackbone(freeze=freeze, normalize=normalize, lora_rank=lora_rank)
    return FoundationALPNetEncoder(backbone=bb, feat_index=feat_index, out_ch=out_ch)


def biomedclip_qnet_encoder(freeze=True, normalize=True, fine_index=None,
                            coarse_index=None, reduce_ch=512, lora_rank=0):
    bb = BiomedCLIPBackbone(freeze=freeze, normalize=normalize, lora_rank=lora_rank)
    return FoundationQNetEncoder(backbone=bb, fine_index=fine_index,
                                 coarse_index=coarse_index, reduce_ch=reduce_ch)
