import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data.dataloader.dataset import EpisodeDataset, get_fold_ids


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
