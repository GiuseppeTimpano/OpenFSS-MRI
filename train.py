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
            data_dir   = self.data_dir,
            scan_ids   = val_ids,
            n_shot     = self.n_shot,
            n_episodes = self.n_val_episodes,
            use_gt     = True,
            augment    = False,
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

        pred_align = self(q_img.unsqueeze(1), q_mask.unsqueeze(1), s_imgs[:, 0])
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

        loss = compute_celoss(pred, q_mask)

        pred_bin = pred.argmax(dim=1).float()
        q_mask_f = q_mask.float()
        inter = (pred_bin * q_mask_f).sum()
        dice  = 2 * inter / (pred_bin.sum() + q_mask_f.sum() + 1e-8)

        self.log_dict({'val/loss': loss, 'val/dice': dice}, on_epoch=True)


if __name__ == '__main__':
    import argparse
    from models.fewshot import FewShotConfig

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     type=str,   required=True)
    parser.add_argument('--model',        type=str,   default='qnet', choices=['qnet', 'alpnet'])
    parser.add_argument('--fold',         type=int,   default=0)
    parser.add_argument('--n_shot',       type=int,   default=1)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--align_weight', type=float, default=1.0)
    parser.add_argument('--batch_size',   type=int,   default=2)
    parser.add_argument('--max_epochs',   type=int,   default=10)
    parser.add_argument('--num_workers',  type=int,   default=4)
    args = parser.parse_args()

    cfg = FewShotConfig(encoder_type=args.model, n_shot=args.n_shot)
    if args.model == 'qnet':
        model = QNetFewShot(cfg)
        align_weight = args.align_weight
    else:
        model = ALPNetFewShot(cfg)
        align_weight = 0.5

    module = FewShotModule(model=model, lr=args.lr, align_weight=align_weight)

    datamodule = FewShotDataModule(
        data_dir    = args.data_dir,
        fold        = args.fold,
        n_shot      = args.n_shot,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
    )

    trainer = pl.Trainer(max_epochs=args.max_epochs)
    trainer.fit(module, datamodule)
