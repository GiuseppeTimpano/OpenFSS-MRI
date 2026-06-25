"""
CycleGAN building blocks for medical MRI domain adaptation.
Generator: 7-layer U-Net with instance norm and Tanh output.
Discriminator: 3-layer PatchGAN with instance norm (1-channel input).
Adapted from medical-I2I-benchmark (src/t1t2converter/models.py):
  - Generator copied unchanged.
  - Discriminator: input_nc changed 2→1 (CycleGAN does not concatenate domains).
"""

import functools

import torch
import torch.nn as nn


def _norm_layer():
    return functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'Norm' in classname:
        if getattr(m, 'weight', None) is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if getattr(m, 'bias', None) is not None:
            nn.init.constant_(m.bias.data, 0.0)


#  U-Net building block 

class _UNetBlock(nn.Module):
    """Recursive U-Net block (innermost / outermost / middle)."""

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=None,
        use_dropout: bool = False,
    ):
        super().__init__()
        self.outermost = outermost
        if norm_layer is None:
            norm_layer = _norm_layer()
        if input_nc is None:
            input_nc = outer_nc

        down_conv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=False)
        down_relu = nn.LeakyReLU(0.2, inplace=True)
        down_norm = norm_layer(inner_nc)
        up_relu   = nn.ReLU(inplace=True)
        up_norm   = norm_layer(outer_nc)

        if outermost:
            up_conv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model   = [down_conv, submodule, up_relu, up_conv, nn.Tanh()]
        elif innermost:
            up_conv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            model   = [down_relu, down_conv, up_relu, up_conv, up_norm]
        else:
            up_conv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=False)
            down    = [down_relu, down_conv, down_norm]
            up      = [up_relu, up_conv, up_norm]
            model   = down + [submodule] + up + ([nn.Dropout(0.5)] if use_dropout else [])

        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


#  Generator 

class UNetGenerator2D(nn.Module):
    """
    7-layer U-Net generator (1→1 channel, tanh output, instance norm).
    Min input size: 256×256 (2^7).
    """

    def __init__(
        self,
        input_nc: int = 1,
        output_nc: int = 1,
        num_downs: int = 7,
        ngf: int = 64,
        use_dropout: bool = False,
    ):
        super().__init__()
        nl = _norm_layer()
        # Build from innermost outward
        blk = _UNetBlock(ngf * 8, ngf * 8, norm_layer=nl, innermost=True)
        for _ in range(num_downs - 5):
            blk = _UNetBlock(ngf * 8, ngf * 8, submodule=blk, norm_layer=nl,
                             use_dropout=use_dropout)
        blk = _UNetBlock(ngf * 4, ngf * 8, submodule=blk, norm_layer=nl)
        blk = _UNetBlock(ngf * 2, ngf * 4, submodule=blk, norm_layer=nl)
        blk = _UNetBlock(ngf,     ngf * 2, submodule=blk, norm_layer=nl)
        self.model = _UNetBlock(output_nc, ngf, input_nc=input_nc, submodule=blk,
                                outermost=True, norm_layer=nl)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ── Discriminator ─────────────────────────────────────────────────────────────

class NLayerDiscriminator2D(nn.Module):
    """
    PatchGAN discriminator (3 layers, instance norm).
    input_nc=1: single-domain input (CycleGAN, no domain concatenation).
    Source benchmark had input_nc=2 (pix2pix paired concatenation) — changed here.
    """

    def __init__(self, input_nc: int = 1, ndf: int = 64, n_layers: int = 3):
        super().__init__()
        nl  = _norm_layer()
        seq = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf      = min(nf * 2, 512)
            seq += [
                nn.Conv2d(nf_prev, nf, kernel_size=4, stride=2, padding=1, bias=False),
                nl(nf),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        nf_prev = nf
        nf = min(nf * 2, 512)
        seq += [
            nn.Conv2d(nf_prev, nf, kernel_size=4, stride=1, padding=1, bias=False),
            nl(nf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 1, kernel_size=4, stride=1, padding=1),
        ]
        self.model = nn.Sequential(*seq)
        self.apply(weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
