"""
LoRA via PEFT for the foundation backbones.

inject_adapter_in_model edits the model in place, keeping the backbone's custom
methods (get_intermediate_layers, etc). Regime B: call AFTER freezing the base;
PEFT adds trainable adapters while the frozen base stays frozen.

Targets are Linear layers (PEFT LoRA is clean on Linear):
  - vit      : attention qkv (fused Q,K,V).
  - convnext : pointwise pwconv1/pwconv2 (the 1x1 convs are nn.Linear in DINOv3
               and hold most of the block's params). The depthwise dwconv is
               grouped (groups = channels) and not adapted here.
"""
from peft import LoraConfig, inject_adapter_in_model


_TARGETS = {
    'vit':      ['qkv'],                  # fused QKV linear (timm + DINOv3)
    'convnext': ['pwconv1', 'pwconv2'],   # pointwise Linear in each ConvNeXt block
}


def apply_lora(module, arch, rank=4, alpha=None, target_modules=None):
    """Inject LoRA adapters into `module` in place and return it."""
    targets = target_modules if target_modules is not None else _TARGETS[arch]
    cfg = LoraConfig(r=rank, lora_alpha=alpha or rank, target_modules=targets,
                     lora_dropout=0.0, bias='none')
    inject_adapter_in_model(cfg, module)
    return module
