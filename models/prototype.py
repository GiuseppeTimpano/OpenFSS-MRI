import numpy as np
import torch
import torch.nn as nn
import torch.nn.parameter as Parameter
import torch.nn.functional as F

from abc import abstractmethod
from typing import Optional

class BasePrototype(nn.Module):

    def __init__(self, eps: float = 1e-4, temperature: float = 20.0):
        super().__init__()
        self.eps = eps
        self.temperature = temperature

    def safe_norm(self, x: torch.Tensor, p: int = 2, dim: int = 1) -> torch.Tensor:
        x_norm = torch.norm(x, p=p, dim=dim)  # [N]
        x_norm = torch.max(x_norm, torch.ones_like(x_norm) * self.eps)
        return x.div(x_norm.unsqueeze(dim).expand_as(x))

    def compute_similarity(self, qry_n: torch.Tensor, proto_n: torch.Tensor) -> torch.Tensor:
        # proto_n: [N, C] as conv 1x1 filters
        # qry_n: [1, C, H, W]
        # output: [1, N, H, W]
        filters = proto_n.unsqueeze(-1).unsqueeze(-1)  # [N, C, 1, 1]
        return F.conv2d(qry_n, filters) * self.temperature

    def aggregate(self, dists: torch.Tensor) -> torch.Tensor:
        # dists: [1, N, H, W]
        # output: [1, 1, H, W]
        weights = F.softmax(dists, dim=1)
        return torch.sum(weights * dists, dim=1, keepdim=True)

    @abstractmethod #abstract method implemented by specific prototype class
    def build_prototype(self, sup_x: torch.Tensor, sup_y: torch.Tensor) -> torch.Tensor:
        # sup_x: [1, C, H, W]
        # sup_y: [1, 1, H, W]
        # return [N, C]
        raise NotImplementedError

    def forward(self, qry: torch.Tensor, sup_x: torch.Tensor, sup_y: torch.Tensor) -> torch.Tensor:
        # qry: [1, C, H, W]
        # sup_x: [1, C, H, W]
        # sup_y: [1, 1, H, W]
        protos = self.build_prototype(sup_x, sup_y)
        proto_n = self.safe_norm(protos)
        qry_n = self.safe_norm(qry)
        dists = self.compute_similarity(qry_n, proto_n)
        pred = self.aggregate(dists)
        return pred


class GlobalPrototype(BasePrototype):

    def __init__(self, eps: float = 1e-4, temperature: float = 20.0):
        super().__init__(eps=eps, temperature=temperature)

    def build_prototype(self, sup_x: torch.Tensor, sup_y: torch.Tensor) -> torch.Tensor:
        # sup_x: [1, C, H, W]
        # sup_y: [1, 1, H, W]
        # masked average pooling → single global prototype
        # output: [1, C]
        proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + self.eps)
        return proto  # [1, C]


class GridPrototype(BasePrototype):

    def __init__(self, proto_grid: list, feature_hw: list, thresh: float = 0.95,
                 eps: float = 1e-4, temperature: float = 20.0,
                 val_pool_size: Optional[int] = None):
        super().__init__(eps=eps, temperature=temperature)
        self.thresh = thresh
        self.val_pool_size = val_pool_size
        kernel_size = [ft // gr for ft, gr in zip(feature_hw, proto_grid)]
        self.pool_op = nn.AvgPool2d(kernel_size)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        # use dynamic pooling at inference when val_pool_size is set
        if not self.training and self.val_pool_size is not None:
            return F.avg_pool2d(x, self.val_pool_size)
        return self.pool_op(x)

    def build_prototype(self, sup_x: torch.Tensor, sup_y: torch.Tensor) -> torch.Tensor:
        # sup_x: [1, C, H, W]
        # sup_y: [1, 1, H, W]
        # output: [N_valid, C]

        C = sup_x.shape[1]

        # step 1: pool feature map → grid candidates
        n_sup_x = self._pool(sup_x)            # [1, C, G, G]
        n_sup_x = n_sup_x.view(1, C, -1)       # [1, C, G*G]
        n_sup_x = n_sup_x.permute(0, 2, 1)     # [1, G*G, C]
        n_sup_x = n_sup_x.squeeze(0)           # [G*G, C]

        # step 2: pool mask → score per cell
        sup_y_g = self._pool(sup_y)            # [1, 1, G, G]
        sup_y_g = sup_y_g.view(-1)             # [G*G]

        # step 3: keep only cells with enough foreground
        protos = n_sup_x[sup_y_g > self.thresh]  # [N_valid, C]

        return protos


class GridPlusPrototype(BasePrototype):

    def __init__(self, proto_grid: list, feature_hw: list, thresh: float = 0.95,
                 eps: float = 1e-4, temperature: float = 20.0,
                 val_pool_size: Optional[int] = None):
        super().__init__(eps=eps, temperature=temperature)
        self.thresh = thresh
        self.val_pool_size = val_pool_size
        kernel_size = [ft // gr for ft, gr in zip(feature_hw, proto_grid)]
        self.pool_op = nn.AvgPool2d(kernel_size)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self.val_pool_size is not None:
            return F.avg_pool2d(x, self.val_pool_size)
        return self.pool_op(x)

    def build_prototype(self, sup_x: torch.Tensor, sup_y: torch.Tensor) -> torch.Tensor:
        # sup_x: [1, C, H, W]
        # sup_y: [1, 1, H, W]
        # output: [N_valid + 1, C]

        C = sup_x.shape[1]

        # local prototypes (same as GridPrototype)
        n_sup_x = self._pool(sup_x)            # [1, C, G, G]
        n_sup_x = n_sup_x.view(1, C, -1)       # [1, C, G*G]
        n_sup_x = n_sup_x.permute(0, 2, 1)     # [1, G*G, C]
        n_sup_x = n_sup_x.squeeze(0)           # [G*G, C]

        sup_y_g = self._pool(sup_y)            # [1, 1, G, G]
        sup_y_g = sup_y_g.view(-1)             # [G*G]

        local_protos = n_sup_x[sup_y_g > self.thresh]  # [N_valid, C]

        # global prototype
        glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) / (sup_y.sum(dim=(-1, -2)) + self.eps)  # [1, C]

        return torch.cat([local_protos, glb_proto], dim=0)  # [N_valid + 1, C]
