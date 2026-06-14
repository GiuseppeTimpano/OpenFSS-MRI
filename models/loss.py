import torch
import torch.nn as nn
from torch.nn import functional as F


def compute_celoss(pred, query_mask):
    '''
        pred: [B, 2, H, W] float - logits
        query_mask: [B, H, W] int - binary mask
    '''
    return F.cross_entropy(input=pred, target=query_mask.long())


def prototype_refinement(
    qry_feat: torch.Tensor,
    proto: torch.Tensor,
    pred: torch.Tensor,
    tao: torch.Tensor,
    n_iters: int = 3,
    lr: float = 0.01,
    temperature: float = 20.0,
) -> torch.Tensor:
    """
    Test-time prototype refinement (Q-Net style). Call only during inference.

    Minimizes BCE between original query features and features reconstructed
    using the prototype, iteratively pulling the prototype toward the actual
    query fg feature distribution.

    qry_feat:    [1, C, h, w]  query feature map
    proto:       [1, C]        initial prototype (from support)
    pred:        [1, 1, h, w]  initial fg probability in [0, 1]
    tao:         scalar        adaptive threshold from encoder
    n_iters:     number of refinement steps
    returns:     [1, C]        refined prototype
    """
    proto_ = nn.Parameter(proto.clone())
    optimizer = torch.optim.Adam([proto_], lr=lr)
    bce = nn.BCELoss()

    def _norm(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((x - x.min()) / (x.max() - x.min() + 1e-8))

    for _ in range(n_iters):
        with torch.enable_grad():
            pred_mask = (pred > 0.5).float().expand_as(qry_feat)
            fg_fts  = proto_.unsqueeze(-1).unsqueeze(-1).expand_as(qry_feat) * pred_mask
            bg_fts  = qry_feat * (1 - pred_mask)
            new_fts = bg_fts + fg_fts
            loss = bce(_norm(new_fts), _norm(qry_feat.detach()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            sim  = -F.cosine_similarity(qry_feat, proto_.unsqueeze(-1).unsqueeze(-1),
                                        dim=1, eps=1e-8) * temperature
            pred = (1.0 - torch.sigmoid(0.5 * (sim - tao))).unsqueeze(1)

    return proto_.detach()
