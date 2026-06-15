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
    encoder_type:     str   = 'qnet'
    n_shot:           int   = 1
    proto_grid:       list  = field(default_factory=lambda: [8, 8])
    feature_hw:       list  = field(default_factory=lambda: [32, 32])
    temperature:      float = 20.0
    refinement_iters: int   = 3    # QNet test-time proto refinement steps (0 = disabled)
    val_wsize:        int   = 4    # ALPNet inference pooling window (None = same as training)

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
        sup_m = F.interpolate(sup_masks.view(B * K, 1, H, W).float(),
                              size=(h, w), mode='nearest').view(B, K, 1, h, w)

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
        align_loss = compute_celoss(align_pred, support_masks[:, 0].long())
        return pred, align_loss


class QNetFewShot(BaseFewShot):

    def __init__(self, cfg):
        super().__init__(cfg)
        # one GlobalPrototype per scale (independent temperature params if needed)
        self.proto_32 = GlobalPrototype(temperature=cfg.temperature)
        self.proto_64 = GlobalPrototype(temperature=cfg.temperature)
        # learnable weights to combine the two scale predictions
        self.alpha = nn.Parameter(torch.ones(2))

    def _predict(self, qry_f32, qry_f64, tao, sup_f32, sup_f64, sup_masks, out_hw):
        # qry_f32:  [B, 512, h32, w32]    target features (finer scale)
        # qry_f64:  [B, 512, h64, w64]    target features (coarser scale)
        # tao:      [B, 1]                adaptive threshold per target image
        # sup_f32:  [B, K, 512, h32, w32] support features (finer scale)
        # sup_f64:  [B, K, 512, h64, w64] support features (coarser scale)
        # sup_masks:[B, K, H, W]          support fg masks (full res, binary)
        # out_hw:   (H, W)
        # output:   [B, 2, H, W]
        B, K, H, W = sup_masks.shape
        h32, w32 = qry_f32.shape[-2:]
        h64, w64 = qry_f64.shape[-2:]

        # resize masks to match each feature scale
        sup_m32 = F.interpolate(sup_masks.view(B * K, 1, H, W).float(),
                                size=(h32, w32), mode='nearest').view(B, K, 1, h32, w32)
        sup_m64 = F.interpolate(sup_masks.view(B * K, 1, H, W).float(),
                                size=(h64, w64), mode='nearest').view(B, K, 1, h64, w64)

        alpha = F.softmax(self.alpha, dim=0)   # normalized combination weights

        preds = []
        for b in range(B):
            qf32 = qry_f32[b:b+1]   # [1, 512, h32, w32]
            qf64 = qry_f64[b:b+1]   # [1, 512, h64, w64]
            t    = tao[b]            # [1] — adaptive threshold for this target image

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

    def forward(self, support_imgs, support_masks, query_img, train=False):
        # support_imgs:  [B, K, H, W]
        # support_masks: [B, K, H, W]  (fg=1, bg=0)
        # query_img:     [B, H, W]
        # output:        [B, 2, H, W]  (ch0=bg logit, ch1=fg logit)
        #                or (pred, align_loss) when train=True
        B, K, H, W = support_imgs.shape

        # encoder returns (feature_dict, tao); keep tao for both sides for alignment
        sup_feats, sup_tao = self.encoder(self._to_3ch(support_imgs.view(B * K, H, W)))
        qry_feats, qry_tao = self.encoder(self._to_3ch(query_img))   # tao: [B, 1], adaptive per image

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
        tgt_tao = sup_tao.view(B, K, 1)[:, 0]   # [B, 1] — threshold of support shot 0
        align_pred = self._predict(
            sup_f32[:, 0], sup_f64[:, 0], tgt_tao,
            qry_f32.unsqueeze(1), qry_f64.unsqueeze(1), pred_bin, (H, W))
        align_loss = compute_celoss(align_pred, support_masks[:, 0].long())
        return pred, align_loss