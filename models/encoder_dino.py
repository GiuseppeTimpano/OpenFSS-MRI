"""
DINOv3 encoders for the prototypical FSS heads (ALPNet / Q-Net).

Goal: keep the prototypical pipeline FIXED and swap only the encoder, so the
backbone becomes a controlled variable for the T1<->T2 robustness study.

Two backbone forms are supported, selected by `arch`:
  - 'vit'      : single-resolution patch tokens (dense map at /patch_size).
                 No native multi-scale -> Q-Net's coarser scale is obtained by
                 average-pooling a deeper-block feature (pseudo-pyramid).
                 Supports LoRA-style adaptation later (Q,V), since it is a ViT.
  - 'convnext' : native hierarchical stages (/4 /8 /16 /32) -> true multi-scale,
                 a natural fit for Q-Net. No LoRA (not attention-based).

Interface parity with models/encoder.py:
  - DINOv3ALPNetEncoder.forward(x) -> [B, 256, h, w]
  - DINOv3QNetEncoder.forward(x)   -> ({'down2':[B,512,hf,wf],
                                        'down3':[B,512,hc,wc]}, tao[B,1])

Weights note: DINOv3 weights are gated. Pass a local hub checkout via
`repo_dir` + `weights` (a local .pth), or use DINOv2 names with the public hub.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import apply_lora


# ImageNet statistics — DINO backbones are trained with this normalization.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _load_dino_backbone(model_name, weights=None, repo_dir=None, source='github'):
    """Load a DINOv2/v3 backbone from torch.hub.

    repo_dir set  -> local checkout  (source='local'), `weights` is a local path.
    repo_dir None -> remote hub 'facebookresearch/dinov3' (weights gated upstream).
    """
    if repo_dir is not None:
        return torch.hub.load(repo_dir, model_name, source='local', weights=weights)
    return torch.hub.load('facebookresearch/dinov3', model_name,
                          source=source, weights=weights)


class DINOv3Backbone(nn.Module):
    """Thin wrapper exposing a uniform feature API over ViT / ConvNeXt DINO models.

    Provides:
      - extract_levels(x, indices) -> list of [B, C, h, w] (one per requested level)
      - global_vector(x)           -> [B, C] pooled descriptor (for Q-Net tao)
      - self.level_dims            -> channel dim of every available level
      - self.global_dim            -> channel dim of the global descriptor
    """

    def __init__(self, arch='vit', model_name='dinov3_vitb16', weights=None,
                 repo_dir=None, source='github', freeze=True, normalize=True,
                 lora_rank=0):
        super().__init__()
        assert arch in ('vit', 'convnext'), f"unknown arch '{arch}'"
        self.arch = arch
        self.normalize = normalize

        self.backbone = _load_dino_backbone(model_name, weights, repo_dir, source)

        if freeze:
            self.backbone.eval()
            for p in self.backbone.parameters():
                p.requires_grad = False
        self._frozen = freeze

        # LoRA (regime B): base stays frozen, PEFT adds trainable adapters. Run the
        # forward under grad so the adapter params receive gradients.
        if lora_rank > 0:
            apply_lora(self.backbone, arch, rank=lora_rank)
            self._frozen = False

        # ImageNet normalization applied inside the encoder (input is RGB-expanded
        # grayscale in [0,1]-ish range from the data pipeline).
        self.register_buffer('_mean', torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('_std', torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1))

        # patch size (ViT) — input H,W must be a multiple of it.
        self.patch_size = int(getattr(self.backbone, 'patch_size', 16)) if arch == 'vit' else None

        # discover channel dims by a dummy forward (robust to arch variants).
        self.level_dims, self.global_dim, self.n_levels = self._probe_dims()

    # normalization
    def _prep(self, x):
        if self.normalize:
            x = (x - self._mean.to(x.dtype)) / self._std.to(x.dtype)
        return x

    # feature extraction
    def _vit_levels(self, x, indices):
        # get_intermediate_layers with reshape=True -> tuple of [B, C, h, w]
        feats = self.backbone.get_intermediate_layers(
            x, n=indices, reshape=True)
        return list(feats)

    def _convnext_levels(self, x, indices):
        # iterate downsample_layers + stages (DINOv3 ConvNeXt layout) -> /4 /8 /16 /32
        downs = self.backbone.downsample_layers
        stages = self.backbone.stages
        feats = []
        out = x
        for i in range(len(stages)):
            out = downs[i](out)
            out = stages[i](out)
            feats.append(out)
        return [feats[i] for i in indices]

    def extract_levels(self, x, indices):
        """Return a list of [B, C, h, w] feature maps for the requested levels.

        indices: ViT -> block indices (e.g. [8, 11]); ConvNeXt -> stage indices 0..3.
        """
        x = self._prep(x)
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            if self.arch == 'vit':
                return self._vit_levels(x, list(indices))
            return self._convnext_levels(x, list(indices))

    def global_vector(self, x):
        """[B, C] descriptor for the adaptive threshold (Q-Net tao)."""
        x = self._prep(x)
        ctx = torch.no_grad() if self._frozen else torch.enable_grad()
        with ctx:
            if self.arch == 'vit':
                feat = self.backbone.forward_features(x)
                if isinstance(feat, dict) and 'x_norm_clstoken' in feat:
                    return feat['x_norm_clstoken']                       # [B, C]
                # fallback: GAP over the last patch map
                lvl = self._vit_levels(x, [self._last_vit_index()])[0]
                return lvl.mean(dim=(2, 3))
            feats = self._convnext_levels(x, [self.n_levels - 1])
            return feats[0].mean(dim=(2, 3))                             # [B, C]

    # ---- dim probing ---------------------------------------------------
    def _last_vit_index(self):
        n_blocks = len(getattr(self.backbone, 'blocks', range(12)))
        return n_blocks - 1

    @torch.no_grad()
    def _probe_dims(self):
        was_training = self.backbone.training
        self.backbone.eval()
        size = 224 if self.arch == 'vit' else 64  # 224 is multiple of common patch sizes
        if self.arch == 'vit' and self.patch_size:
            size = (224 // self.patch_size) * self.patch_size
        dummy = torch.zeros(1, 3, size, size)
        dummy = (dummy - self._mean) / self._std if self.normalize else dummy

        if self.arch == 'vit':
            n_blocks = len(getattr(self.backbone, 'blocks', range(12)))
            feats = self._vit_levels(dummy, list(range(n_blocks)))
            level_dims = [f.shape[1] for f in feats]
            gvec = self.global_vector(torch.zeros(1, 3, size, size))
            global_dim = gvec.shape[1]
            n_levels = n_blocks
        else:
            feats = self._convnext_levels(dummy, [0, 1, 2, 3])
            level_dims = [f.shape[1] for f in feats]
            global_dim = level_dims[-1]
            n_levels = len(level_dims)

        if was_training:
            self.backbone.train()
        return level_dims, global_dim, n_levels


class FoundationALPNetEncoder(nn.Module):
    """ALPNet-compatible: single 256ch feature map [B, 256, h, w].

    Pass a prebuilt `backbone` (DINOv3Backbone or BiomedCLIPBackbone). If None,
    a DINOv3Backbone is built from the arch/model_name args.

    feat_index selects which backbone level feeds the head:
      - ViT: a block index (default = last block, richest semantics).
      - ConvNeXt: a stage index 0..3 (default = 2, i.e. /16 — semantic but not /32).
    """

    def __init__(self, backbone=None, arch='vit', model_name='dinov3_vitb16',
                 weights=None, repo_dir=None, source='github', freeze=True,
                 normalize=True, feat_index=None, out_ch=256, lora_rank=0):
        super().__init__()
        if backbone is None:
            backbone = DINOv3Backbone(arch, model_name, weights, repo_dir,
                                      source, freeze, normalize, lora_rank)
        self.backbone = backbone
        if feat_index is None:
            feat_index = (backbone.n_levels - 1) if backbone.arch == 'vit' else 2
        self.feat_index = feat_index
        in_ch = backbone.level_dims[feat_index]

        self.localconv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.localconv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: [B, 3, H, W]  ->  [B, 256, h, w]
        feat = self.backbone.extract_levels(x, [self.feat_index])[0]
        return self.localconv(feat)


class FoundationQNetEncoder(nn.Module):
    """Q-Net-compatible: two 512ch scales (down2 finer, down3 coarser) + tao.

    Pass a prebuilt `backbone` (DINOv3Backbone or BiomedCLIPBackbone). If None,
    a DINOv3Backbone is built from the arch/model_name args.

    Scale selection:
      - ConvNeXt: native stages. Defaults to (/8, /16) = stage indices (1, 2),
        a 2x ratio mirroring ResNet Q-Net's (/4, /8).
      - ViT: single resolution. down2 = a deeper block at /patch; down3 = a (later)
        block average-pooled by 2 -> pseudo-coarser scale (/2*patch).
    """

    def __init__(self, backbone=None, arch='vit', model_name='dinov3_vitb16',
                 weights=None, repo_dir=None, source='github', freeze=True,
                 normalize=True, fine_index=None, coarse_index=None, reduce_ch=512,
                 lora_rank=0):
        super().__init__()
        if backbone is None:
            backbone = DINOv3Backbone(arch, model_name, weights, repo_dir,
                                      source, freeze, normalize, lora_rank)
        self.backbone = backbone
        self.arch = backbone.arch

        if backbone.arch == 'convnext':
            self.fine_index = 1 if fine_index is None else fine_index
            self.coarse_index = 2 if coarse_index is None else coarse_index
            self.vit_pool_coarse = False
        else:
            n = self.backbone.n_levels
            # two deeper blocks; coarser one gets an extra 2x avg-pool
            self.fine_index = (n - 4) if fine_index is None else fine_index
            self.coarse_index = (n - 1) if coarse_index is None else coarse_index
            self.vit_pool_coarse = True

        cf = self.backbone.level_dims[self.fine_index]
        cc = self.backbone.level_dims[self.coarse_index]

        self.reduce1 = nn.Conv2d(cf, reduce_ch, kernel_size=1, bias=False)  # down2 (finer)
        self.reduce2 = nn.Conv2d(cc, reduce_ch, kernel_size=1, bias=False)  # down3 (coarser)
        # tao from a global descriptor of the deepest feature (GAP of the coarse
        # level), mirroring original Q-Net's fc(avgpool(layer4)) — no extra pass.
        self.reduce1d = nn.Linear(cc, 1, bias=True)                         # tao
        nn.init.kaiming_normal_(self.reduce1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.reduce2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.reduce1d.weight)
        nn.init.constant_(self.reduce1d.bias, 0)

    def forward(self, x):
        # x: [B, 3, H, W]  ->  ({'down2','down3'}, tao[B,1])
        fine, coarse = self.backbone.extract_levels(x, [self.fine_index, self.coarse_index])
        if self.vit_pool_coarse:
            coarse = F.avg_pool2d(coarse, kernel_size=2)   # pseudo-coarser scale

        down2 = self.reduce1(fine)      # [B, 512, hf, wf]
        down3 = self.reduce2(coarse)    # [B, 512, hc, wc]

        gvec = coarse.mean(dim=(2, 3))  # [B, Cc] — GAP of deepest feature
        tao = self.reduce1d(gvec)       # [B, 1]
        return {'down2': down2, 'down3': down3}, tao


# back-compat names (heads now accept any backbone, not just DINO)
DINOv3ALPNetEncoder = FoundationALPNetEncoder
DINOv3QNetEncoder = FoundationQNetEncoder
