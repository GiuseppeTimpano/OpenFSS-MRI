import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ALPNetEncoder, QNetEncoder
from .prototype import GlobalPrototype, GridPlusPrototype, GridPrototype
from dataclasses import dataclass, field
from abc import abstractmethod
from .loss import prototype_refinement


@dataclass
class FewShotConfig():
    fg_thresh:        float = 0.95
    bg_thresh:        float = 0.05
    pretrained:       bool  = True
    encoder_type:     str   = 'qnet'
    n_shot:           int   = 1
    proto_grid:       list  = field(default_factory=lambda: [8, 8])
    feature_hw:       list  = field(default_factory=lambda: [32, 32])
    temperature:      float = 20.0
    refinement_iters: int   = 3    # QNet test-time proto refinement steps (0 = disabled)

class BaseFewShot(nn.Module):
    def __init__(self, cfg: FewShotConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.build_encoder()

    def build_encoder(self):
        if self.cfg.encoder_type == 'qnet':
            return QNetEncoder(pretrained=self.cfg.pretrained)
        return ALPNetEncoder(pretrained=self.cfg.pretrained)

    @staticmethod
    def _to_3ch(x: torch.Tensor) -> torch.Tensor:
        # [B, H, W] grayscale → [B, 3, H, W] RGB
        return x.unsqueeze(1).expand(-1, 3, -1, -1).float()

    @abstractmethod
    def forward(self, support_imgs, support_masks, query_img):
        raise NotImplementedError


class ALPNetFewShot(BaseFewShot):

    def __init__(self, cfg: FewShotConfig):
        super().__init__(cfg)
        # fg: gridconv+ (local+global), fallback to global when coverage too low
        self.fg_prototype = GridPlusPrototype(
            cfg.proto_grid, cfg.feature_hw, cfg.fg_thresh, 1e-4, cfg.temperature)
        self.fg_fallback  = GlobalPrototype(temperature=cfg.temperature)
        # bg: gridconv (local only), uses lower bg_thresh
        self.bg_prototype = GridPrototype(
            cfg.proto_grid, cfg.feature_hw, cfg.bg_thresh, 1e-4, cfg.temperature)

    def forward(self, support_imgs, support_masks, query_img):
        # support_imgs:  [B, K, H, W]
        # support_masks: [B, K, H, W]  (fg=1, bg=0)
        # query_img:     [B, H, W]
        # output:        [B, 2, H, W]  (ch0=bg logit, ch1=fg logit)
        B, K, H, W = support_imgs.shape

        sup_feat  = self.encoder(self._to_3ch(support_imgs.view(B * K, H, W)))  # [B*K, 256, h, w]
        h, w      = sup_feat.shape[-2:]

        sup_masks = support_masks.view(B * K, 1, H, W).float()
        sup_masks = F.interpolate(sup_masks, size=(h, w), mode='nearest')        # [B*K, 1, h, w]

        qry_feat  = self.encoder(self._to_3ch(query_img))                        # [B, 256, h, w]

        sup_feat  = sup_feat.view(B, K, -1, h, w)   # [B, K, 256, h, w]
        sup_masks = sup_masks.view(B, K, 1, h, w)   # [B, K, 1, h, w]

        preds = []
        for b in range(B):
            qf = qry_feat[b:b+1]   # [1, 256, h, w]
            shot_preds = []
            for k in range(K):
                fg_msk = sup_masks[b, k:k+1]   # [1, 1, h, w]
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


class QNetFewShot(BaseFewShot):

    def __init__(self, cfg):
        super().__init__(cfg)
        # one GlobalPrototype per scale (independent temperature params if needed)
        self.proto_32 = GlobalPrototype(temperature=cfg.temperature)
        self.proto_64 = GlobalPrototype(temperature=cfg.temperature)
        # learnable weights to combine the two scale predictions
        self.alpha = nn.Parameter(torch.ones(2))

    def forward(self, support_imgs, support_masks, query_img):
        # support_imgs:  [B, K, H, W]
        # support_masks: [B, K, H, W]  (fg=1, bg=0)
        # query_img:     [B, H, W]
        # output:        [B, 2, H, W]  (ch0=bg logit, ch1=fg logit)
        B, K, H, W = support_imgs.shape

        # encoder returns (feature_dict, tao)
        sup_feats, _   = self.encoder(self._to_3ch(support_imgs.view(B * K, H, W)))
        qry_feats, tao = self.encoder(self._to_3ch(query_img))   # tao: [B, 1], adaptive per image

        sup_f32 = sup_feats['down2']   # [B*K, 512, H/4, W/4]
        sup_f64 = sup_feats['down3']   # [B*K, 512, H/8, W/8]
        qry_f32 = qry_feats['down2']   # [B,   512, H/4, W/4]
        qry_f64 = qry_feats['down3']   # [B,   512, H/8, W/8]

        h32, w32 = sup_f32.shape[-2:]
        h64, w64 = sup_f64.shape[-2:]

        # resize masks to match each feature scale
        sup_m32 = F.interpolate(support_masks.view(B * K, 1, H, W).float(),
                                size=(h32, w32), mode='nearest')          # [B*K, 1, h32, w32]
        sup_m64 = F.interpolate(support_masks.view(B * K, 1, H, W).float(),
                                size=(h64, w64), mode='nearest')          # [B*K, 1, h64, w64]

        sup_f32 = sup_f32.view(B, K, -1, h32, w32)
        sup_f64 = sup_f64.view(B, K, -1, h64, w64)
        sup_m32 = sup_m32.view(B, K, 1, h32, w32)
        sup_m64 = sup_m64.view(B, K, 1, h64, w64)

        alpha = F.softmax(self.alpha, dim=0)   # normalized combination weights

        preds = []
        for b in range(B):
            qf32 = qry_f32[b:b+1]   # [1, 512, h32, w32]
            qf64 = qry_f64[b:b+1]   # [1, 512, h64, w64]
            t    = tao[b]            # [1] — adaptive threshold for this query image

            shot_preds = []
            for k in range(K):
                sf32 = sup_f32[b, k:k+1]   # [1, 512, h32, w32]
                sf64 = sup_f64[b, k:k+1]   # [1, 512, h64, w64]
                sm32 = sup_m32[b, k:k+1]   # [1, 1, h32, w32]
                sm64 = sup_m64[b, k:k+1]   # [1, 1, h64, w64]

                sim32 = self.proto_32(qf32, sf32, sm32)   # [1, 1, h32, w32]
                sim64 = self.proto_64(qf64, sf64, sm64)   # [1, 1, h64, w64]

                # upsample coarser scale to match finer scale, then combine
                sim64_up = F.interpolate(sim64, size=(h32, w32), mode='bilinear', align_corners=True)
                fg_sim   = alpha[0] * sim32 + alpha[1] * sim64_up   # [1, 1, h32, w32]

                # test-time prototype refinement (inference only)
                n_iters = self.cfg.refinement_iters
                if not self.training and n_iters > 0:
                    # extract initial prototype [1, C] and initial fg probability [1, 1, h32, w32]
                    proto32 = self.proto_32.build_prototype(sf32, sm32)          # [1, 512]
                    pred_prob = torch.softmax(
                        torch.cat([t.view(1, 1, 1, 1).expand_as(fg_sim), fg_sim], dim=1), dim=1
                    )[:, 1:2]                                                    # [1, 1, h32, w32]

                    proto32_r = prototype_refinement(
                        qf32, proto32, pred_prob, t,
                        n_iters=n_iters,
                        temperature=self.cfg.temperature,
                    )   # [1, 512]

                    # recompute sim32 with refined prototype
                    proto_n = self.proto_32.safe_norm(proto32_r)    # [1, 512]
                    qf32_n  = self.proto_32.safe_norm(qf32)         # [1, 512, h32, w32]
                    sim32_r = self.proto_32.compute_similarity(qf32_n, proto_n)  # [1, 1, h32, w32]
                    sim32_r = self.proto_32.aggregate(sim32_r)                   # [1, 1, h32, w32]
                    fg_sim  = alpha[0] * sim32_r + alpha[1] * sim64_up

                # tao as adaptive bg logit: fg wins where sim > tao, bg wins where sim < tao
                tao_map = t.view(1, 1, 1, 1).expand_as(fg_sim)
                shot_preds.append(torch.cat([tao_map, fg_sim], dim=1))  # [1, 2, h32, w32]

            preds.append(torch.stack(shot_preds).mean(dim=0))   # [1, 2, h32, w32]

        pred = torch.cat(preds, dim=0)                           # [B, 2, h32, w32]
        return F.interpolate(pred, size=(H, W), mode='bilinear', align_corners=True)