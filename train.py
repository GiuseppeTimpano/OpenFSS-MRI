import os
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
        batch_size: int = 2,
        num_workers: int = 4,
        # min foreground pixels per training episode (original min_size filter)
        min_size: int = 200,
        # GT labels to exclude from the SSL training pool (original exclude_label; None = off)
        exclude_label: list[int] | None = None,
    ):
        super().__init__()
        self.data_dir         = data_dir
        self.fold             = fold
        self.n_folds          = n_folds
        self.n_shot           = n_shot
        self.n_train_episodes = n_train_episodes
        self.batch_size       = batch_size
        self.num_workers      = num_workers
        self.min_size         = min_size
        self.exclude_label    = exclude_label

    def setup(self, stage=None):
        # Original Q-Net / SSL-ALPNet protocol: NO validation during training.
        # fold split only separates train scans from the held-out test scans;
        # test scans are evaluated volumetrically by a separate test script.
        train_ids, _ = get_fold_ids(self.data_dir, self.fold, self.n_folds)

        self.train_ds = EpisodeDataset(
            data_dir   = self.data_dir,
            scan_ids   = train_ids,
            n_shot     = self.n_shot,
            n_episodes = self.n_train_episodes,
            use_gt        = False,
            augment       = True,
            min_size      = self.min_size,
            exclude_label = self.exclude_label,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size  = self.batch_size,
            shuffle     = False,
            num_workers = self.num_workers,
            pin_memory  = True,
        )


class FewShotModule(pl.LightningModule):
    def __init__(self, model, lr: float, align_weight: float = 1.0, lr_gamma: float = 0.95):
        super().__init__()
        self._model = model
        self.lr = lr
        self.align_weight = align_weight
        self.lr_gamma = lr_gamma

    def forward(self, support_imgs, support_masks, query_img, train=False):
        return self._model(support_imgs, support_masks, query_img, train=train)

    def configure_optimizers(self):
        # Faithful to original Q-Net / SSL-ALPNet: SGD + MultiStepLR.
        #   optim = {lr: 1e-3, momentum: 0.9, weight_decay: 5e-4}
        #   lr_milestones = [(ii+1)*1000 for ii in range(n_steps//1000 - 1)]
        #   lr_step_gamma = 0.95   (scheduler.step() every optimizer step)
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4,
        )
        total_steps = int(self.trainer.estimated_stepping_batches)
        milestones  = [(ii + 1) * 1000 for ii in range(total_steps // 1000)]
        scheduler   = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=self.lr_gamma,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }

    def training_step(self, batch, batch_idx):
        s_imgs  = batch['support_imgs']   # [B, K, H, W]
        s_masks = batch['support_masks']  # [B, K, H, W]
        q_img   = batch['query_img']      # [B, H, W]
        q_mask  = batch['query_mask']     # [B, H, W]

        # alignment loss reuses encoder features (computed inside forward), no re-encoding
        pred, loss_align = self(s_imgs, s_masks, q_img, train=True)
        # query loss is model-specific (QNet: NLL on probabilities; ALPNet: weighted CE on logits)
        loss = self._model.query_loss(pred, q_mask)

        total = loss + self.align_weight * loss_align
        self.log_dict({'train/loss': loss, 'train/loss_align': loss_align, 'train/total': total}, prog_bar=True)
        return total


def train_from_cfg(cfg: dict) -> str:
    """Train model from cfg dict. Returns path to last.ckpt."""
    from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
    from pytorch_lightning.loggers import CSVLogger
    from models.fewshot import FewShotConfig

    data_cfg   = cfg['data']
    model_cfg  = cfg['model']
    model_name = model_cfg['name']
    train_cfg  = cfg['train']

    fcfg = FewShotConfig(
        encoder_type = model_name,
        n_shot       = data_cfg['n_shot'],
    )
    bg_loss_weight = train_cfg.get('bg_loss_weight', 0.1)
    model          = QNetFewShot(fcfg, bg_loss_weight=bg_loss_weight) \
                     if model_name == 'qnet' \
                     else ALPNetFewShot(fcfg, bg_loss_weight=bg_loss_weight)

    module = FewShotModule(
        model        = model,
        lr           = train_cfg['lr'],
        align_weight = train_cfg['align_weight'],
        lr_gamma     = train_cfg.get('lr_gamma', 0.95),
    )

    datamodule = FewShotDataModule(
        data_dir      = data_cfg['data_dir'],
        fold          = data_cfg['fold'],
        n_folds       = data_cfg['n_folds'],
        n_shot        = data_cfg['n_shot'],
        batch_size    = data_cfg['batch_size'],
        num_workers   = data_cfg['num_workers'],
        min_size      = data_cfg.get('min_size', 200),
        exclude_label = data_cfg.get('exclude_label'),
    )

    modality  = next((m for m in ('T1', 'T2') if m in data_cfg['data_dir']), 'MRI')
    setting   = 's2' if data_cfg.get('exclude_label') else 's1'
    run_name  = f"{model_name}_resnet_{modality}_fold{data_cfg['fold']}_{setting}"

    logger = CSVLogger('.', name='lightning_logs', version=run_name)
    os.makedirs(logger.log_dir, exist_ok=True)

    trainer = pl.Trainer(
        max_epochs = train_cfg['max_epochs'],
        num_sanity_val_steps = 0,
        callbacks=[
            ModelCheckpoint(save_top_k=-1, every_n_epochs=1, save_last=True,
                            filename='{epoch}-{step}'),
            RichProgressBar(),
        ],
        logger=logger,
    )

    trainer.fit(module, datamodule)
    return os.path.join(logger.log_dir, 'checkpoints', 'last.ckpt')


if __name__ == '__main__':
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/resnet.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    ckpt = train_from_cfg(cfg_file)
    print(f'checkpoint saved → {ckpt}')
