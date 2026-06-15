import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataloader.dataset import EpisodeDataset, get_fold_ids
from models.fewshot import QNetFewShot, ALPNetFewShot
from models.loss import compute_celoss


class FewShotDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        fold: int = 0,
        n_folds: int = 4,
        n_shot: int = 1,
        n_train_episodes: int = 1000,
        n_val_episodes: int = 200,
        batch_size: int = 2,
        num_workers: int = 4,
        # label names for GT classmap decoding (required when use_gt=True)
        label_names: list[str] | None = None,
        # domain-shift options (val only: train always same-domain)
        domain_map: dict[str, str] | None = None,
        source_domain: str | None = None,
        target_domain: str | None = None,
    ):
        super().__init__()
        self.data_dir         = data_dir
        self.fold             = fold
        self.n_folds          = n_folds
        self.n_shot           = n_shot
        self.n_train_episodes = n_train_episodes
        self.n_val_episodes   = n_val_episodes
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.label_names      = label_names
        self.domain_map       = domain_map
        self.source_domain    = source_domain
        self.target_domain    = target_domain

    @property
    def cross_domain(self) -> bool:
        return self.domain_map is not None

    def setup(self, stage=None):
        train_ids, val_ids = get_fold_ids(self.data_dir, self.fold, self.n_folds)

        self.train_ds = EpisodeDataset(
            data_dir   = self.data_dir,
            scan_ids   = train_ids,
            n_shot     = self.n_shot,
            n_episodes = self.n_train_episodes,
            use_gt     = False,
            augment    = True,
        )
        self.val_ds = EpisodeDataset(
            data_dir      = self.data_dir,
            scan_ids      = val_ids,
            n_shot        = self.n_shot,
            n_episodes    = self.n_val_episodes,
            use_gt        = True,
            augment       = False,
            label_names   = self.label_names,
            cross_domain  = self.cross_domain,
            source_domain = self.source_domain,
            target_domain = self.target_domain,
            domain_map    = self.domain_map,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size  = 1,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )


class FewShotModule(pl.LightningModule):
    def __init__(self, model, lr: float, align_weight: float = 1.0):
        super().__init__()
        self._model = model
        self.lr = lr
        self.align_weight = align_weight

    def forward(self, support_imgs, support_masks, query_img):
        return self._model(support_imgs, support_masks, query_img)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        s_imgs  = batch['support_imgs']   # [B, K, H, W]
        s_masks = batch['support_masks']  # [B, K, H, W]
        q_img   = batch['query_img']      # [B, H, W]
        q_mask  = batch['query_mask']     # [B, H, W]

        pred = self(s_imgs, s_masks, q_img)
        loss = compute_celoss(pred, q_mask)

        # use predicted mask (not GT) as query-as-support — matches original ALPNet alignment
        with torch.no_grad():
            pred_bin = pred.argmax(dim=1, keepdim=True).float()  # [B, 1, H, W]
        pred_align = self(q_img.unsqueeze(1), pred_bin, s_imgs[:, 0])
        loss_align = compute_celoss(pred_align, s_masks[:, 0].long())

        total = loss + self.align_weight * loss_align
        self.log_dict({'train/loss': loss, 'train/loss_align': loss_align, 'train/total': total})
        return total

    def validation_step(self, batch, batch_idx):
        s_imgs  = batch['support_imgs']
        s_masks = batch['support_masks']
        q_img   = batch['query_img']
        q_mask  = batch['query_mask']

        pred = self(s_imgs, s_masks, q_img)

        loss     = compute_celoss(pred, q_mask)
        pred_bin = pred.argmax(dim=1).float()
        q_mask_f = q_mask.float()
        inter    = (pred_bin * q_mask_f).sum()
        dice     = 2 * inter / (pred_bin.sum() + q_mask_f.sum() + 1e-8)

        # always log aggregate metrics
        self.log_dict({'val/loss': loss, 'val/dice': dice}, on_epoch=True)

        # if cross-domain episode, also log per domain-pair metrics
        # DataLoader collates strings into lists and bools into tensors
        if batch['cross_domain'][0].item():
            src = batch['source_domain'][0]
            tgt = batch['target_domain'][0]
            pair = f'{src}→{tgt}'
            self.log_dict(
                {f'val/loss_{pair}': loss, f'val/dice_{pair}': dice},
                on_epoch=True,
            )


if __name__ == '__main__':
    import argparse
    import json
    import yaml
    from models.fewshot import FewShotConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='path to the YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    data_cfg   = cfg_file['data']
    model_name = cfg_file['model']['name']
    train_cfg  = cfg_file['train']
    domain_cfg = cfg_file.get('domain', {})

    # load domain map if a path is given
    domain_map = None
    if domain_cfg.get('domain_map'):
        with open(domain_cfg['domain_map']) as f:
            domain_map = json.load(f)

    cfg = FewShotConfig(encoder_type=model_name, n_shot=data_cfg['n_shot'])
    if model_name == 'qnet':
        model = QNetFewShot(cfg)
        align_weight = train_cfg['align_weight']
    else:
        model = ALPNetFewShot(cfg)
        align_weight = 0.5

    module = FewShotModule(model=model, lr=train_cfg['lr'], align_weight=align_weight)

    datamodule = FewShotDataModule(
        data_dir      = data_cfg['data_dir'],
        fold          = data_cfg['fold'],
        n_folds       = data_cfg['n_folds'],
        n_shot        = data_cfg['n_shot'],
        batch_size    = data_cfg['batch_size'],
        num_workers   = data_cfg['num_workers'],
        label_names   = data_cfg['label_names'],
        domain_map    = domain_map,
        source_domain = domain_cfg.get('source_domain'),
        target_domain = domain_cfg.get('target_domain'),
    )

    trainer = pl.Trainer(max_epochs=train_cfg['max_epochs'])
    trainer.fit(module, datamodule)
