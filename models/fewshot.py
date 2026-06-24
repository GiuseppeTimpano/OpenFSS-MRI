import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ALPNetEncoder, QNetEncoder
from .prototype import GlobalPrototype, GridPlusPrototype, GridPrototype
from dataclasses import dataclass, field
from abc import abstractmethod
from .loss import prototype_refinement, compute_celoss


@dataclass
class FewShotConfig():
    fg_thresh:        float = 0.95
    bg_thresh:        float = 0.95
    pretrained:       bool  = True
    encoder_type:     str   = 'qnet'   # head shape: 'qnet' (dual-scale) | 'alpnet' (single)
    n_shot:           int   = 1
    proto_grid:       list  = field(default_factory=lambda: [8, 8])
    feature_hw:       list  = field(default_factory=lambda: [32, 32])
    temperature:      float = 20.0
    refinement_iters: int   = 7    # QNet test-time proto refinement steps (0 = disabled)
    val_wsize:        int   = 2    # ALPNet inference pooling window (original SSL-ALPNet test: 2; None = same as training)

class BaseFewShot(nn.Module):
    def __init__(self, cfg: FewShotConfig, bg_loss_weight: float = 0.1):
        super().__init__()
        self.cfg = cfg
        self.bg_loss_weight = bg_loss_weight
        self.encoder = self.build_encoder()

    def query_loss(self, pred, mask):
        # default (ALPNet): raw-similarity logits → weighted CE, faithful to SSL-ALPNet
        # (CrossEntropyLoss(weight=[bg, 1.0])).
        weight = torch.tensor([self.bg_loss_weight, 1.0], device=pred.device)
        return compute_celoss(pred, mask.long(), weight=weight)

    def align_loss_fn(self, pred, mask):
        # original align loss is UNWEIGHTED (both ALPNet CrossEntropy and Q-Net NLLLoss).
        return compute_celoss(pred, mask.long(), weight=None)

    @staticmethod
    def _nll(prob, mask, weight=None):
        # NLLLoss on log-probabilities (Q-Net: pred is a probability, not a logit).
        eps  = torch.finfo(torch.float32).eps
        logp = torch.log(torch.clamp(prob, eps, 1 - eps))
        return F.nll_loss(logp, mask.long(), weight=weight)

    def build_encoder(self):
        c = self.cfg
        is_qnet = c.encoder_type == 'qnet'
        if is_qnet:
            return QNetEncoder(pretrained=c.pretrained)
        return ALPNetEncoder(pretrained=c.pretrained)

    @staticmethod
    def _to_3ch(x: torch.Tensor) -> torch.Tensor:
        # [B, H, W] grayscale → [B, 3, H, W] RGB
        return x.unsqueeze(1).expand(-1, 3, -1, -1).float()

    @abstractmethod
    def forward(self, support_imgs, support_masks, query_img):
        raise NotImplementedError


class ALPNetFewShot(BaseFewShot):

    def __init__(self, cfg: FewShotConfig, bg_loss_weight: float = 0.05):
        super().__init__(cfg, bg_loss_weight=bg_loss_weight)
        # fg: gridconv+ (local+global), fallback to global when coverage too low
        self.fg_prototype = GridPlusPrototype(
            cfg.proto_grid, cfg.feature_hw, cfg.fg_thresh, 1e-4, cfg.temperature,
            val_pool_size=cfg.val_wsize)
        self.fg_fallback  = GlobalPrototype(temperature=cfg.temperature)
        # bg: gridconv (local only)
        self.bg_prototype = GridPrototype(
            cfg.proto_grid, cfg.feature_hw, cfg.bg_thresh, 1e-4, cfg.temperature,
            val_pool_size=cfg.val_wsize)

    def _predict(self, qry_feat, sup_feat, sup_masks, out_hw):
        # qry_feat:  [B, 256, h, w]      target features to segment
        # sup_feat:  [B, K, 256, h, w]   support features
        # sup_masks: [B, K, H, W]        support fg masks (full res, binary)
        # out_hw:    (H, W)
        # output:    [B, 2, H, W]
        B, K, H, W = sup_masks.shape
        h, w = qry_feat.shape[-2:]
        # original SSL-ALPNet downsamples the support mask to feature res with BILINEAR
        # (grid_proto_fewshot.py: F.interpolate(fore_mask, fts_size, mode='bilinear')),
        # giving soft mask values used for grid-cell thresholding + global prototype.
        sup_m = F.interpolate(sup_masks.view(B * K, 1, H, W).float(),
                              size=(h, w), mode='bilinear', align_corners=False).view(B, K, 1, h, w)

        preds = []
        for b in range(B):
            qf = qry_feat[b:b+1]   # [1, 256, h, w]
            shot_preds = []
            for k in range(K):
                fg_msk = sup_m[b, k:k+1]   # [1, 1, h, w]
                bg_msk = 1.0 - fg_msk

                # if fg coverage too low, gridconv+ yields no valid cells → fallback to global
                coverage = F.avg_pool2d(fg_msk, 4).max()
                if coverage >= self.cfg.fg_thresh:
                    fg_score = self.fg_prototype(qf, sup_feat[b, k:k+1], fg_msk)
                else:
                    fg_score = self.fg_fallback(qf, sup_feat[b, k:k+1], fg_msk)

                bg_score = self.bg_prototype(qf, sup_feat[b, k:k+1], bg_msk)

                shot_preds.append(torch.cat([bg_score, fg_score], dim=1))  # [1, 2, h, w]
            preds.append(torch.stack(shot_preds).mean(dim=0))              # [1, 2, h, w]

        pred = torch.cat(preds, dim=0)                                     # [B, 2, h, w]
        return F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)

    def forward(self, support_imgs, support_masks, query_img, train=False):
        # support_imgs:  [B, K, H, W]
        # support_masks: [B, K, H, W]  (fg=1, bg=0)
        # query_img:     [B, H, W]
        # output:        [B, 2, H, W]  (ch0=bg logit, ch1=fg logit)
        #                or (pred, align_loss) when train=True
        B, K, H, W = support_imgs.shape

        sup_feat = self.encoder(self._to_3ch(support_imgs.view(B * K, H, W)))  # [B*K, 256, h, w]
        qry_feat = self.encoder(self._to_3ch(query_img))                       # [B, 256, h, w]
        h, w = sup_feat.shape[-2:]
        sup_feat = sup_feat.view(B, K, -1, h, w)                               # [B, K, 256, h, w]

        pred = self._predict(qry_feat, sup_feat, support_masks, (H, W))
        if not train:
            return pred

        # alignment: query-derived prototype predicts on support — reuses encoder
        # features (no second forward pass), matching the original Q-Net alignLoss
        with torch.no_grad():
            pred_bin = pred.argmax(dim=1, keepdim=True).float()   # [B, 1, H, W]
        align_pred = self._predict(sup_feat[:, 0], qry_feat.unsqueeze(1), pred_bin, (H, W))
        align_loss = self.align_loss_fn(align_pred, support_masks[:, 0].long())
        return pred, align_loss


class QNetFewShot(BaseFewShot):

    # fixed scale-combination weights (original Q-Net: alpha=[0.9, 0.1], NOT learnable)
    ALPHA = (0.9, 0.1)

    def __init__(self, cfg, bg_loss_weight: float = 0.1):
        super().__init__(cfg, bg_loss_weight=bg_loss_weight)
        # one GlobalPrototype per scale (masked-average-pool prototype builder)
        self.proto_32 = GlobalPrototype(temperature=cfg.temperature)
        self.proto_64 = GlobalPrototype(temperature=cfg.temperature)

    def query_loss(self, pred, mask):
        # pred is a probability map [B,2,H,W]; original Q-Net query loss = NLLLoss(weight=[0.1,1.0]).
        weight = torch.tensor([self.bg_loss_weight, 1.0], device=pred.device)
        return self._nll(pred, mask, weight=weight)

    def align_loss_fn(self, pred, mask):
        # original Q-Net alignLoss = nn.NLLLoss() (unweighted) on log-probabilities.
        return self._nll(pred, mask, weight=None)

    def _get_pred(self, qry_feat, proto, tao):
        # original getPred: sim = -cos * scaler ;  p_fg = 1 - sigmoid(0.5 * (sim - tao))
        # qry_feat: [1, C, h, w]   proto: [1, C]   tao: [1]   →   [1, 1, h, w] (fg probability)
        sim = -F.cosine_similarity(qry_feat, proto[..., None, None], dim=1, eps=1e-4) * self.cfg.temperature  # [1,h,w]
        p   = 1.0 - torch.sigmoid(0.5 * (sim - tao.view(1, 1, 1)))   # [1, h, w]
        return p.unsqueeze(1)   # [1, 1, h, w]

    def _build_proto(self, proto_mod, sup_feat, sup_mask):
        # masked-avg prototype averaged over shots (original getPrototype: sum_shots / n_shots)
        K = sup_feat.shape[0]
        protos = [proto_mod.build_prototype(sup_feat[k:k+1], sup_mask[k:k+1]) for k in range(K)]
        return torch.stack(protos, dim=0).mean(dim=0)   # [1, C]

    def _predict(self, qry_f32, qry_f64, tao, sup_f32, sup_f64, sup_masks, out_hw):
        # qry_f32:  [B, 512, h32, w32]    target features (finer scale)
        # qry_f64:  [B, 512, h64, w64]    target features (coarser scale)
        # tao:      [B, 1]                adaptive threshold per target image
        # sup_f32:  [B, K, 512, h32, w32] support features (finer scale)
        # sup_f64:  [B, K, 512, h64, w64] support features (coarser scale)
        # sup_masks:[B, K, H, W]          support fg masks (full res, binary)
        # out_hw:   (H, W)
        # output:   [B, 2, H, W]          PROBABILITIES (ch0=bg, ch1=fg)
        B, K, H, W = sup_masks.shape

        # Keep masks at FULL resolution [B, K, 1, H, W]. The prototype builder
        # upsamples the features to this resolution before pooling (original Q-Net
        # getFeatures), instead of shrinking the masks down to the feature grid.
        sup_m = sup_masks.unsqueeze(2).float()   # [B, K, 1, H, W]

        a0, a1 = self.ALPHA   # fixed weights (sum = 1.0)

        preds = []
        for b in range(B):
            qf32 = qry_f32[b:b+1]   # [1, 512, h32, w32]
            qf64 = qry_f64[b:b+1]   # [1, 512, h64, w64]
            t    = tao[b]            # [1] — adaptive threshold for this target image

            # one prototype per scale, averaged over shots (full-res masks)
            proto32 = self._build_proto(self.proto_32, sup_f32[b], sup_m[b])   # [1, C]
            proto64 = self._build_proto(self.proto_64, sup_f64[b], sup_m[b])   # [1, C]

            # test-time prototype refinement (inference only) — original updatePrototype, both scales
            n_iters = self.cfg.refinement_iters
            if not self.training and n_iters > 0:
                proto32 = prototype_refinement(
                    qf32, proto32, self._get_pred(qf32, proto32, t), t,
                    n_iters=n_iters, temperature=self.cfg.temperature)
                proto64 = prototype_refinement(
                    qf64, proto64, self._get_pred(qf64, proto64, t), t,
                    n_iters=n_iters, temperature=self.cfg.temperature)

            # per-scale fg probability (original getPred), interpolate to out size, fixed-alpha combine
            p32 = F.interpolate(self._get_pred(qf32, proto32, t), size=(H, W), mode='bilinear', align_corners=True)
            p64 = F.interpolate(self._get_pred(qf64, proto64, t), size=(H, W), mode='bilinear', align_corners=True)
            p   = a0 * p32 + a1 * p64   # [1, 1, H, W] — probability, sum(alpha)=1
            preds.append(torch.cat([1.0 - p, p], dim=1))   # [1, 2, H, W]

        return torch.cat(preds, dim=0)   # [B, 2, H, W] probabilities

    def forward(self, support_imgs, support_masks, query_img, train=False):
        # support_imgs:  [B, K, H, W]
        # support_masks: [B, K, H, W]  (fg=1, bg=0)
        # query_img:     [B, H, W]
        # output:        [B, 2, H, W]  (ch0=bg logit, ch1=fg logit)
        #                or (pred, align_loss) when train=True
        B, K, H, W = support_imgs.shape

        # encoder returns (feature_dict, tao); original Q-Net uses the QUERY threshold (self.t)
        # for both query and alignment predictions, so the support tao is unused.
        sup_feats, _sup_tao = self.encoder(self._to_3ch(support_imgs.view(B * K, H, W)))
        qry_feats, qry_tao  = self.encoder(self._to_3ch(query_img))   # tao: [B, 1], adaptive per image

        hs32, ws32 = sup_feats['down2'].shape[-2:]
        hs64, ws64 = sup_feats['down3'].shape[-2:]
        sup_f32 = sup_feats['down2'].view(B, K, -1, hs32, ws32)   # [B, K, 512, h32, w32]
        sup_f64 = sup_feats['down3'].view(B, K, -1, hs64, ws64)   # [B, K, 512, h64, w64]
        qry_f32 = qry_feats['down2']   # [B, 512, h32, w32]
        qry_f64 = qry_feats['down3']   # [B, 512, h64, w64]

        pred = self._predict(qry_f32, qry_f64, qry_tao, sup_f32, sup_f64, support_masks, (H, W))
        if not train:
            return pred

        # alignment: query-derived prototype predicts on support — reuses encoder
        # features (no second forward pass), matching the original Q-Net alignLoss
        with torch.no_grad():
            pred_bin = pred.argmax(dim=1, keepdim=True).float()   # [B, 1, H, W]
        # original alignLoss: query-derived prototype predicts on support, using the QUERY tao
        align_pred = self._predict(
            sup_f32[:, 0], sup_f64[:, 0], qry_tao,
            qry_f32.unsqueeze(1), qry_f64.unsqueeze(1), pred_bin, (H, W))
        align_loss = self.align_loss_fn(align_pred, support_masks[:, 0].long())
        return pred, align_loss