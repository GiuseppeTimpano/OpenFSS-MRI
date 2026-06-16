import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights


class _BaseEncoder(nn.Module):
    """Shared backbone loading logic for ResNet101-based encoders."""

    def _build_backbone_from_resnet(self, pretrained, replace_stride_with_dilation):
        _model = models.resnet101(
            weights=None,
            replace_stride_with_dilation=replace_stride_with_dilation
        )
        self.backbone = nn.ModuleDict()
        for name, module in _model.named_children():
            self.backbone[name] = module
        if pretrained:
            ckpt = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1.get_state_dict(progress=True)
            own = self.state_dict()
            for k, v in ckpt.items():
                if k in own and own[k].shape == v.shape:
                    own[k] = v
            self.load_state_dict(own)

    def _build_backbone_from_deeplab(self, pretrained):
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None
        _model = models.segmentation.deeplabv3_resnet101(weights=weights)
        self.backbone = nn.ModuleDict()
        for name, module in _model.backbone.named_children():  # .backbone to skip ASPP
            self.backbone[name] = module

    def _forward_backbone(self, x):
        x = self.backbone['conv1'](x)
        x = self.backbone['bn1'](x)
        x = self.backbone['relu'](x)
        x = self.backbone['maxpool'](x)
        x = self.backbone['layer1'](x)
        x = self.backbone['layer2'](x)
        x = self.backbone['layer3'](x)
        layer3_out = x
        x = self.backbone['layer4'](x)
        layer4_out = x
        return layer3_out, layer4_out


class ALPNetEncoder(_BaseEncoder):
    """
    ALPNet encoder: DeepLab ResNet101 backbone, single 256ch feature map.
    dilation: [False, True, True] — output spatial /8
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self._build_backbone_from_deeplab(pretrained)
        self.localconv = nn.Conv2d(2048, 256, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.localconv.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x: [B, 3, H, W]
        _, layer4_out = self._forward_backbone(x)
        out = self.localconv(layer4_out)    # [B, 256, H/8, W/8]
        return out


class QNetEncoder(_BaseEncoder):
    """
    Q-Net encoder: vanilla ResNet101, two 512ch feature maps + adaptive threshold tao.
    dilation: [True, True, False] — down2 at /4, down3 at /8
    """

    def __init__(self, pretrained=True):
        super().__init__()
        self._build_backbone_from_resnet(pretrained, replace_stride_with_dilation=[True, True, False])
        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)  # layer3 → down2 /4
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)  # layer4 → down3 /8
        self.reduce1d = nn.Linear(1000, 1, bias=True)                   # tao
        self._init_new_layers()

    def _init_new_layers(self):
        nn.init.kaiming_normal_(self.reduce1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.reduce2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.reduce1d.weight)
        nn.init.constant_(self.reduce1d.bias, 0)

    def forward(self, x):
        # x: [B, 3, H, W]
        layer3_out, layer4_out = self._forward_backbone(x)
        down2 = self.reduce1(layer3_out)    # [B, 512, H/4, W/4]
        down3 = self.reduce2(layer4_out)    # [B, 512, H/8, W/8]

        t = self.backbone['avgpool'](layer4_out)    # [B, 2048, 1, 1]
        t = torch.flatten(t, 1)                     # [B, 2048]
        t = self.backbone['fc'](t)                  # [B, 1000]
        tao = self.reduce1d(t)                      # [B, 1]

        return {'down2': down2, 'down3': down3}, tao
