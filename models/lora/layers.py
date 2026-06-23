"""
Custom convolutional LoRA layers (ported from the dafne project).

Kept for ConvNeXt depthwise-conv experiments. NOTE: as-is these build a dense
[C_out, C_in, ...] delta and do not account for grouped/depthwise convs (groups);
they are correct for groups=1 convs. For the foundation backbones the LoRA wiring
uses PEFT (ViT qkv + ConvNeXt pointwise Linear), so these are not wired in by
default.
"""
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


class LoRANdConvLayer(nn.Module):
    '''
    LoRA for generic N-Dimensional Convolutional Layers (1D, 2D, 3D)
    '''

    def __init__(self, base_module: nn.modules.conv._ConvNd, lora_config: dict) -> None:
        '''
        Args:
            base_module (nn.modules.conv._ConvNd): The base convolutional layer to apply LoRA to.
            lora_config (dict): A dictionary containing LoRA configuration parameters.
                Expected keys:
                - 'rank' (int): The rank of the low-rank matrices.
                - 'alpha' (int): The scaling factor for the low-rank update.
        '''
        super(LoRANdConvLayer, self).__init__()

        assert isinstance(base_module, nn.modules.conv._ConvNd), 'base model must be a torch.nn.modules.conv._ConvNd'

        self.base_module = base_module
        for param in self.base_module.parameters():
            param.requires_grad = False

        self.is_transposed = isinstance(base_module, nn.modules.conv._ConvTransposeNd)

        # Consistent channel assignment (ConvTranspose has inverted weight shape: [in, out/groups, ...])
        in_channels = self.base_module.in_channels
        out_channels = self.base_module.out_channels
        kernel_size = self.base_module.kernel_size

        self.nd = len(kernel_size)
        self.kernel_size = kernel_size

        self.register_buffer('alpha',
                            torch.tensor(lora_config['alpha'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device))

        self.register_buffer('rank',
                            torch.tensor(lora_config['rank'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device))

        assert 'rank_for' in lora_config, 'rank_for not found in lora_config'
        assert lora_config['rank_for'] in ['kernel', 'channels'], 'Invalid `rank_for` value!'
        self.rank_for = lora_config['rank_for']

        # Initialize LoRA parameters
        if self.rank_for == 'kernel':
            # Spatial decomposition of the last spatial dimension.
            # 2D: [C_out, C_in, kH, kW] -> A: [C_out, C_in, kH, r], B: [C_out, C_in, r, kW]
            # 3D: [C_out, C_in, kD, kH, kW] -> A: [C_out, C_in, kD, kH, r], B: [C_out, C_in, kD, r, kW]
            spatial_dims = list(kernel_size)
            last_dim = spatial_dims.pop()

            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, *spatial_dims, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            # FIX: B must drop the last *remaining* spatial dim so A and B share batch
            # dims for matmul (the original used *spatial_dims here -> shape mismatch for nd>=2).
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(out_channels, in_channels, *spatial_dims[:-1], lora_config['rank'], last_dim),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        elif self.rank_for == 'channels':
            # Channel decomposition: [C_out, C_in, *kernel] -> A: [*kernel, C_out, r], B: [*kernel, r, C_in]
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(*kernel_size, out_channels, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(*kernel_size, lora_config['rank'], in_channels),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        self.reset_trainable_params()
        self.adapter_enabled = False

    def reset_trainable_params(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5**0.5)
        nn.init.zeros_(self.delta_weight_B)

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            delta_lora = None
            if self.rank_for == 'kernel':
                delta_lora = torch.matmul(self.delta_weight_A, self.delta_weight_B)
                if self.is_transposed:
                    delta_lora = delta_lora.transpose(0, 1)
            elif self.rank_for == 'channels':
                lora_delta = torch.matmul(self.delta_weight_A, self.delta_weight_B)

                if not self.is_transposed:
                    permute_idx = (self.nd, self.nd + 1) + tuple(range(self.nd))
                elif self.is_transposed:
                    permute_idx = (self.nd + 1, self.nd) + tuple(range(self.nd))
                delta_lora = lora_delta.permute(permute_idx)

            delta_lora = (self.alpha / self.rank) * delta_lora

            if self.is_transposed:
                conv_fn = getattr(F, f"conv_transpose{self.nd}d")
                delta_output = conv_fn(
                    input=x,
                    weight=delta_lora,
                    bias=None,
                    stride=self.base_module.stride,
                    padding=self.base_module.padding,
                    dilation=self.base_module.dilation,
                    groups=self.base_module.groups,
                    output_padding=self.base_module.output_padding
                )
            else:
                conv_fn = getattr(F, f"conv{self.nd}d")
                delta_output = conv_fn(
                    input=x,
                    weight=delta_lora,
                    bias=None,
                    stride=self.base_module.stride,
                    padding=self.base_module.padding,
                    dilation=self.base_module.dilation,
                    groups=self.base_module.groups
                )
            return self.base_module(x) + delta_output

        else:
            return self.base_module(x)

    def __repr__(self) -> str:
        spatial_info = ", ".join([f"k{i}={s}" for i, s in enumerate(self.kernel_size)])
        adapter_repr_string = f'Adapter({spatial_info}, rank={self.rank.item()}, rank_for={self.rank_for})'
        return f'LoRANdConvLayer({self.base_module} + α={self.alpha.item()}/r={self.rank.item()} × {adapter_repr_string})'

    def _get_merged_modules(self) -> nn.Module:
        out_channels, in_channels, *kernel_size = self.base_module.weight.size()

        if self.adapter_enabled:
            if self.rank_for == 'kernel':
                delta_lora = torch.matmul(self.delta_weight_A, self.delta_weight_B)
                if self.is_transposed:
                    delta_lora = delta_lora.transpose(0, 1)
            elif self.rank_for == 'channels':
                lora_delta = torch.matmul(self.delta_weight_A, self.delta_weight_B)

                if not self.is_transposed:
                    permute_idx = (self.nd, self.nd + 1) + tuple(range(self.nd))
                elif self.is_transposed:
                    permute_idx = (self.nd + 1, self.nd) + tuple(range(self.nd))
                delta_lora = lora_delta.permute(permute_idx)

            delta_lora = (self.alpha / self.rank) * delta_lora
            new_weights = self.base_module.weight + delta_lora
        else:
            new_weights = self.base_module.weight

        if self.is_transposed:
            conv_cls = getattr(nn, f"ConvTranspose{self.nd}d")
            merged_module = conv_cls(
                in_channels=self.base_module.in_channels,
                out_channels=self.base_module.out_channels,
                kernel_size=self.base_module.kernel_size,
                stride=self.base_module.stride,
                padding=self.base_module.padding,
                dilation=self.base_module.dilation,
                groups=self.base_module.groups,
                bias=(self.base_module.bias is not None),
                output_padding=self.base_module.output_padding
            )
        else:
            conv_cls = getattr(nn, f"Conv{self.nd}d")
            merged_module = conv_cls(
                in_channels=self.base_module.in_channels,
                out_channels=self.base_module.out_channels,
                kernel_size=self.base_module.kernel_size,
                stride=self.base_module.stride,
                padding=self.base_module.padding,
                dilation=self.base_module.dilation,
                groups=self.base_module.groups,
                bias=(self.base_module.bias is not None),
                padding_mode=self.base_module.padding_mode
            )
        merged_module.weight.data = new_weights.data
        if self.base_module.bias is not None:
            merged_module.bias.data = self.base_module.bias.data
        return merged_module


class LoRA2dConvLayer(nn.Module):
    '''
    LoRA for 2D Convolutional Layers
    '''
    def __init__(self, base_module: nn.Conv2d, lora_config:dict) -> None:
        '''
        Args:
            base_module (nn.Conv2d): The base convolutional layer to apply LoRA to.
            lora_config (dict): A dictionary containing LoRA configuration parameters.
                Expected keys:
                - 'rank' (int): The rank of the low-rank matrices.
                - 'alpha' (int): The scaling factor for the low-rank update.
        '''
        super(LoRA2dConvLayer, self).__init__()

        assert isinstance(base_module, nn.Conv2d), 'base model must be torch.nn.Conv2d'

        self.base_module = base_module
        for param in self.base_module.parameters():
            param.requires_grad = False #extract W and b and freeze them

        out_channels, in_channels, kH, kW = self.base_module.weight.size() #get dimensions of the base module

        # initialize lora params as non trainable params in the forward pass
        self.register_buffer('alpha',
                            torch.tensor(lora_config['alpha'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device)) #add alpha to register buffer

        self.register_buffer('rank',
                            torch.tensor(lora_config['rank'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device)) #add rank to register buffer

        assert 'rank_for' in lora_config.keys(), 'rank_for not found in lora_config'
        assert lora_config['rank_for'] in ['kernel', 'channels'], 'Invalid `rank_for` value! Please pick from the valid values: "kernel", "channels".'
        self.rank_for = lora_config['rank_for']

        #initialize delta matrix for lora: it depends on the rank_for parameter

        #spatial decomposition. filter dimension: [C_out, C_in, kH, kW].
        #A dimension: [C_out, C_in, kH, r]. B dimension: [C_out, C_in, r, kW]
        if self.rank_for == 'kernel':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size = (out_channels, in_channels, kH, lora_config['rank']),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size = (out_channels, in_channels, lora_config['rank'], kW),
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        #channel decomposition. filter dimension: [C_out, C_in, kH, kW].
        elif self.rank_for == 'channels':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(kH, kW, out_channels, lora_config['rank']), #A dimension: [C_out, r, kH, kW]
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(kH, kW, lora_config['rank'], in_channels), #B dimension: [r, C_in, kH, kW]
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
        self.reset_trainable_params()
        self.adapter_enabled = False #controls the inferencing, "base" or "base + adapter"

    def reset_trainable_params(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5**0.5)
        nn.init.zeros_(self.delta_weight_B)


    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            if self.rank_for == 'kernel':
                delta_weight = torch.matmul(self.delta_weight_A, self.delta_weight_B)
            elif self.rank_for == 'channels':
                delta_weight = torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1)

            delta_output = F.conv2d(
                input=x,
                weight=delta_weight,
                bias=None,
                stride=self.base_module.stride,
                padding=self.base_module.padding,
                dilation=self.base_module.dilation,
                groups=self.base_module.groups
            )
            return self.base_module(x) + (self.alpha / self.rank) * delta_output

        return self.base_module(x)

    #get lora layer representation
    def __repr__(self) -> str:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            adapter_repr_string = f'Adapter(kH={kH}, rank={self.rank.item()}, kW={kW})'
        elif self.rank_for == 'channels':
            adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_features={out_channels})'

        repr_string = f'LoRAConv2d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        return repr_string

    def _get_merged_modules(self) -> nn.Conv2d:
        out_channels, in_channels, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            new_weights = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
        elif self.rank_for == 'channels':
            new_weights = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(2, 3, 0, 1))
        bias = self.base_module.bias

        # FIX: original passed these positionally and misaligned groups/bias/padding_mode.
        merged_module = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kH, kW),
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups,
            bias=bias is not None,
            padding_mode=self.base_module.padding_mode,
        )
        merged_module.weight.data = new_weights
        if bias is not None:
            merged_module.bias.data = bias.data
        return merged_module


class LoRA3dConvLayer(nn.Module):
    '''
    LoRA for 3D Convolutional Layers
    '''
    def __init__(self, base_module: nn.Conv3d, lora_config:dict) -> None:
        '''
        Args:
            base_module (nn.Conv3d): The base convolutional layer to apply LoRA to.
            lora_config (dict): A dictionary containing LoRA configuration parameters.
                Expected keys:
                - 'rank' (int): The rank of the low-rank matrices.
                - 'alpha' (int): The scaling factor for the low-rank update.
        '''
        super(LoRA3dConvLayer, self).__init__()

        assert isinstance(base_module, nn.Conv3d), 'base model must be torch.nn.Conv3d'

        self.base_module = base_module
        for param in self.base_module.parameters():
            param.requires_grad = False #extract W and b and freeze them

        out_channels, in_channels, kD, kH, kW = self.base_module.weight.size() #get dimensions of the base module

        # initialize lora params as non trainable params in the forward pass
        self.register_buffer('alpha',
                            torch.tensor(lora_config['alpha'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device)) #add alpha to register buffer

        self.register_buffer('rank',
                            torch.tensor(lora_config['rank'],
                            dtype=self.base_module.weight.dtype,
                            device=self.base_module.weight.device)) #add rank to register buffer

        assert 'rank_for' in lora_config.keys(), 'rank_for not found in lora_config'
        assert lora_config['rank_for'] in ['kernel', 'channels'], 'Invalid `rank_for` value! Please pick from the valid values: "kernel", "channels".'
        self.rank_for = lora_config['rank_for']

        #initialize delta matrix for lora: it depends on the rank_for parameter

        #spatial decomposition. filter dimension: [C_out, C_in, kD, kH, kW].
        # B dimension: [C_out, C_in, r, kH, kW]
        if self.rank_for == 'kernel':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size = (out_channels, in_channels, kD, kH, lora_config['rank']), # A dimension: [C_out, C_in, kD, kH, r], D as dimension of batch
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size = (out_channels, in_channels, kD, lora_config['rank'], kW), # B dimension: [C_out, C_in, kD, r, kW]
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )

        #channel decomposition. filter dimension: [C_out, C_in, kH, kW].
        elif self.rank_for == 'channels':
            self.delta_weight_A = nn.Parameter(
                torch.empty(
                    size=(kD, kH, kW, out_channels, lora_config['rank']), #A dimension: [kD, kH, kW, C_out, r]
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
            self.delta_weight_B = nn.Parameter(
                torch.empty(
                    size=(kD, kH, kW, lora_config['rank'], in_channels), #B dimension: [kD, kH, kW, r, C_in]
                    dtype=self.base_module.weight.dtype,
                    device=self.base_module.weight.device
                )
            )
        self.reset_trainable_params()
        self.adapter_enabled = False #controls the inferencing, "base" or "base + adapter"

    def reset_trainable_params(self) -> None:
        nn.init.kaiming_uniform_(self.delta_weight_A, a=5**0.5)
        nn.init.zeros_(self.delta_weight_B)

    def enable_adapter(self) -> None:
        self.adapter_enabled = True

    def disable_adapter(self) -> None:
        self.adapter_enabled = False

    def forward(self, x: Tensor) -> Tensor:
        if self.adapter_enabled:
            if self.rank_for == 'kernel':
                delta_weight = torch.matmul(self.delta_weight_A, self.delta_weight_B)
            elif self.rank_for == 'channels':
                delta_weight = torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(3, 4, 0, 1, 2)

            delta_output = F.conv3d(
                input=x,
                weight=delta_weight,
                bias=None,
                stride=self.base_module.stride,
                padding=self.base_module.padding,
                dilation=self.base_module.dilation,
                groups=self.base_module.groups
            )
            return self.base_module(x) + (self.alpha / self.rank) * delta_output

        return self.base_module(x)

    #get lora layer representation
    def __repr__(self) -> str:
        out_channels, in_channels, kD, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            adapter_repr_string = f'Adapter(kD={kD}, kH={kH}, kW={kW}, rank={self.rank.item()})'
        elif self.rank_for == 'channels':
            adapter_repr_string = f'Adapter(in_channels={in_channels}, rank={self.rank.item()}, out_channels={out_channels})'

        repr_string = f'LoRAConv3d({self.base_module} + ((α={self.alpha.item()}/r={self.rank.item()}) × {adapter_repr_string}))'
        return repr_string

    def _get_merged_modules(self) -> nn.Conv3d:
        out_channels, in_channels, kD, kH, kW = self.base_module.weight.size()

        if self.rank_for == 'kernel':
            new_weights = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B))
        elif self.rank_for == 'channels':
            new_weights = self.base_module.weight + ((self.alpha / self.rank) * torch.matmul(self.delta_weight_A, self.delta_weight_B).permute(3, 4, 0, 1, 2))
        bias = self.base_module.bias

        # FIX: original passed these positionally and misaligned groups/bias/padding_mode.
        merged_module = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kD, kH, kW),
            stride=self.base_module.stride,
            padding=self.base_module.padding,
            dilation=self.base_module.dilation,
            groups=self.base_module.groups,
            bias=bias is not None,
            padding_mode=self.base_module.padding_mode,
        )
        merged_module.weight.data = new_weights
        if bias is not None:
            merged_module.bias.data = bias.data
        return merged_module
