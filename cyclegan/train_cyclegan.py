"""
CycleGAN training (PyTorch Lightning) for medical MRI domain adaptation.

T2→T1:
  python -m cyclegan.train_cyclegan \
      --src_dir data/datasets/CHAOS/processed/T2 \
      --tgt_dir data/datasets/CHAOS/processed/T1 \
      --out_dir cyclegan/runs/T2_to_T1

AMOS→CHAOS:
  python -m cyclegan.train_cyclegan \
      --src_dir data/datasets/AMOS/processed/T2 \
      --tgt_dir data/datasets/CHAOS/processed/T2 \
      --out_dir cyclegan/runs/AMOS_to_CHAOS
"""

import argparse
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from cyclegan.models import UNetGenerator2D, NLayerDiscriminator2D
from cyclegan.dataset import UnpairedNIfTIDataset


class _ReplayBuffer:
    def __init__(self, max_size: int = 50):
        self.max_size = max_size
        self.data: list[torch.Tensor] = []

    def push_and_pop(self, batch: torch.Tensor) -> torch.Tensor:
        result = []
        for img in batch:
            img = img.unsqueeze(0)
            if len(self.data) < self.max_size:
                self.data.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                result.append(self.data[idx].clone())
                self.data[idx] = img
            else:
                result.append(img)
        return torch.cat(result, dim=0)


class CycleGAN(L.LightningModule):
    def __init__(self, lr=2e-4, lambda_cyc=10.0, lambda_id=5.0, lambda_organ=0.0,
                 organ_std=False, epochs=200, stats_A=None, stats_B=None):
        super().__init__()
        self.save_hyperparameters(ignore=['stats_A', 'stats_B'])
        self.automatic_optimization = False

        # per-domain organ intensity priors (mean, std) in [-1,1]; used by region loss
        self.stats_A = stats_A or {}
        self.stats_B = stats_B or {}

        self.G_AB = UNetGenerator2D()
        self.G_BA = UNetGenerator2D()
        self.D_A  = NLayerDiscriminator2D()
        self.D_B  = NLayerDiscriminator2D()

        self.crit_adv = nn.MSELoss()  # LSGAN: linear D output, no sigmoid (CycleGAN default)
        self.crit_cyc = nn.L1Loss()
        self.crit_id  = nn.L1Loss()

        self.buf_A = _ReplayBuffer()
        self.buf_B = _ReplayBuffer()

    def configure_optimizers(self):
        decay_start = self.hparams.epochs // 2

        def lr_lambda(epoch):
            if epoch < decay_start:
                return 1.0
            return 1.0 - (epoch - decay_start) / max(1, self.hparams.epochs - decay_start)

        opt_G   = torch.optim.Adam(
            list(self.G_AB.parameters()) + list(self.G_BA.parameters()),
            lr=self.hparams.lr, betas=(0.5, 0.999),
        )
        opt_D_A = torch.optim.Adam(self.D_A.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_D_B = torch.optim.Adam(self.D_B.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))

        schedulers = [
            {"scheduler": torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda), "interval": "epoch"}
            for opt in [opt_G, opt_D_A, opt_D_B]
        ]
        return [opt_G, opt_D_A, opt_D_B], schedulers

    def training_step(self, batch, batch_idx):
        real_A, real_B, lab_A, lab_B = batch
        opt_G, opt_D_A, opt_D_B = self.optimizers()

        # Generator step (D frozen)
        self._set_requires_grad([self.D_A, self.D_B], False)
        opt_G.zero_grad()

        fake_B = self.G_AB(real_A)
        fake_A = self.G_BA(real_B)
        rec_A  = self.G_BA(fake_B)
        rec_B  = self.G_AB(fake_A)

        loss_adv = (
            self.crit_adv(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B))) +
            self.crit_adv(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
        )
        loss_cyc = (self.crit_cyc(rec_A, real_A) + self.crit_cyc(rec_B, real_B)) * self.hparams.lambda_cyc
        loss_id  = (
            self.crit_id(self.G_BA(real_A), real_A) +
            self.crit_id(self.G_AB(real_B), real_B)
        ) * self.hparams.lambda_id
        # region-aware organ loss: push each organ in the fake toward the TARGET
        # domain's known intensity stats (fake_B uses A's mask + B's stats, & symm.)
        loss_organ = (
            self._organ_loss(fake_B, lab_A, self.stats_B) +
            self._organ_loss(fake_A, lab_B, self.stats_A)
        ) * self.hparams.lambda_organ
        loss_G = loss_adv + loss_cyc + loss_id + loss_organ

        self.manual_backward(loss_G)
        opt_G.step()

        # Discriminator step (replay buffer for stability)
        self._set_requires_grad([self.D_A, self.D_B], True)

        opt_D_A.zero_grad()
        pred_real_A = self.D_A(real_A)
        pred_fake_A = self.D_A(self.buf_A.push_and_pop(fake_A.detach()))
        loss_D_A = (
            self.crit_adv(pred_real_A, torch.ones_like(pred_real_A) * 0.9) +
            self.crit_adv(pred_fake_A, torch.zeros_like(pred_fake_A))
        ) * 0.5
        self.manual_backward(loss_D_A)
        opt_D_A.step()

        opt_D_B.zero_grad()
        pred_real_B = self.D_B(real_B)
        pred_fake_B = self.D_B(self.buf_B.push_and_pop(fake_B.detach()))
        loss_D_B = (
            self.crit_adv(pred_real_B, torch.ones_like(pred_real_B) * 0.9) +
            self.crit_adv(pred_fake_B, torch.zeros_like(pred_fake_B))
        ) * 0.5
        self.manual_backward(loss_D_B)
        opt_D_B.step()

        self.log_dict({
            "loss_G": loss_G, "loss_adv": loss_adv,
            "loss_cyc": loss_cyc, "loss_id": loss_id, "loss_organ": loss_organ,
            "loss_D_A": loss_D_A, "loss_D_B": loss_D_B,
        }, on_step=False, on_epoch=True, prog_bar=True)

        if (self.current_epoch + 1) % 10 == 0 and batch_idx == 0:
            self._save_samples(real_A, real_B, fake_B)

    def _organ_loss(self, fake, lab, stats):
        """L1 match of fake intensity to target-domain stats, per organ region
        defined by the source mask `lab`. Pools per class across the batch.
        Mean term only by default; the std term (organ_std=True) tends to push
        pixels toward [-1,1] extremes — clashing with the adversarial texture —
        and was the cause of the black voids/blotches. Returns 0 if no stats."""
        loss = fake.new_zeros(())
        if not stats:
            return loss
        for k, (mu, sd) in stats.items():
            m = lab == k
            if m.any():
                vals = fake[m]
                loss = loss + (vals.mean() - mu).abs()
                if self.hparams.organ_std:
                    loss = loss + (vals.std(unbiased=False) - sd).abs()
        return loss

    @staticmethod
    def _set_requires_grad(nets, flag):
        for net in nets:
            for p in net.parameters():
                p.requires_grad = flag

    def _save_samples(self, real_A, real_B, fake_B):
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            out_dir = self.logger.log_dir if self.logger else "."
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            for ax, img, title in zip(
                axes,
                [real_A[0, 0], real_B[0, 0], fake_B[0, 0]],
                ['real_A', 'real_B', 'fake_B'],
            ):
                ax.imshow(img.detach().cpu().numpy(), cmap='gray', vmin=-1, vmax=1)
                ax.set_title(title, fontsize=8); ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'sample_ep{self.current_epoch + 1:03d}.png'), dpi=100)
            plt.close()
        except Exception as e:
            print(f'[sample save skipped: {e}]')


def _plot_losses(out_dir: str):
    import glob
    import pandas as pd
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    csv_files = glob.glob(os.path.join(out_dir, 'logs', '**', 'metrics.csv'), recursive=True)
    if not csv_files:
        print('No metrics.csv found, skipping loss plot.')
        return

    df = pd.read_csv(csv_files[0]).dropna(subset=['epoch'])
    df = df.groupby('epoch').mean(numeric_only=True).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(df['epoch'], df['loss_G'],   label='loss_G')
    axes[0].plot(df['epoch'], df['loss_adv'], label='loss_adv', linestyle='--')
    axes[0].plot(df['epoch'], df['loss_cyc'], label='loss_cyc', linestyle='--')
    axes[0].plot(df['epoch'], df['loss_id'],  label='loss_id',  linestyle='--')
    if 'loss_organ' in df.columns:
        axes[0].plot(df['epoch'], df['loss_organ'], label='loss_organ', linestyle=':')
    axes[0].set_title('Generator losses'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df['epoch'], df['loss_D_A'], label='loss_D_A')
    axes[1].plot(df['epoch'], df['loss_D_B'], label='loss_D_B')
    axes[1].set_title('Discriminator losses'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f'Loss curves saved → {plot_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',          required=True)
    parser.add_argument('--tgt_dir',          required=True)
    parser.add_argument('--src_manifest',     default=None,
                        help='scanner_manifest.json for src_dir (enables scanner filtering)')
    parser.add_argument('--src_manufacturer', default=None,
                        help='filter src cases by manufacturer substring (e.g. SIEMENS)')
    parser.add_argument('--src_model',        default=None,
                        help='filter src cases by model substring (e.g. Prisma)')
    parser.add_argument('--tgt_manifest',     default=None,
                        help='scanner_manifest.json for tgt_dir')
    parser.add_argument('--tgt_manufacturer', default=None,
                        help='filter tgt cases by manufacturer substring')
    parser.add_argument('--tgt_model',        default=None,
                        help='filter tgt cases by model substring')
    parser.add_argument('--out_dir',    default='cyclegan/runs')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--lambda_cyc', type=float, default=10.0)
    parser.add_argument('--lambda_id',  type=float, default=2.5)
    parser.add_argument('--lambda_organ', type=float, default=0.0,
                        help='region-aware organ intensity loss weight (0=off). Try 5')
    parser.add_argument('--organ_std', action='store_true',
                        help='also match per-organ intensity std (default: mean only). '
                             'std term causes black voids/blotches; leave off')
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--pair_mode',  default='auto',
                        choices=['auto', 'subject', 'depth', 'random'],
                        help='B-slice pairing: auto (subject if ids overlap else depth)')
    parser.add_argument('--depth_tol',  type=float, default=0.1,
                        help='depth window half-width (fraction of volume depth)')
    parser.add_argument('--min_body',   type=float, default=0.05,
                        help='drop slices with body fraction below this (near-black)')
    parser.add_argument('--no_aug',     action='store_true',
                        help='disable train-time augmentation (hflip + small affine)')
    parser.add_argument('--save_ckpt',  action='store_true',
                        help='Save full Lightning checkpoint (default: only generators saved)')
    parser.add_argument('--device',     default='auto')
    args = parser.parse_args()

    case_ids_A = None
    if args.src_manifest:
        case_ids_A = UnpairedNIfTIDataset.case_ids_from_manifest(
            args.src_manifest, args.src_manufacturer, args.src_model)
        print(f'src filter → {len(case_ids_A)} cases: {case_ids_A}')

    case_ids_B = None
    if args.tgt_manifest:
        case_ids_B = UnpairedNIfTIDataset.case_ids_from_manifest(
            args.tgt_manifest, args.tgt_manufacturer, args.tgt_model)
        print(f'tgt filter → {len(case_ids_B)} cases: {case_ids_B}')

    dataset = UnpairedNIfTIDataset(args.src_dir, args.tgt_dir,
                                   min_body=args.min_body,
                                   pair_mode=args.pair_mode, depth_tol=args.depth_tol,
                                   augment=not args.no_aug,
                                   case_ids_A=case_ids_A, case_ids_B=case_ids_B)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    model = CycleGAN(lr=args.lr, lambda_cyc=args.lambda_cyc,
                     lambda_id=args.lambda_id, lambda_organ=args.lambda_organ,
                     organ_std=args.organ_std, epochs=args.epochs,
                     stats_A=dataset.stats_A, stats_B=dataset.stats_B)

    callbacks = [RichProgressBar()]
    if args.save_ckpt:
        callbacks.append(ModelCheckpoint(
            dirpath=os.path.join(args.out_dir, 'checkpoints'),
            filename='cyclegan_final', save_top_k=1, monitor=None,
        ))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.device,
        default_root_dir=args.out_dir,
        logger=CSVLogger(save_dir=args.out_dir, name='logs'),
        callbacks=callbacks,
        enable_checkpointing=args.save_ckpt,
        log_every_n_steps=1,
    )
    trainer.fit(model, loader)

    final_path = os.path.join(args.out_dir, 'generators_final.pth')
    torch.save({'G_AB': model.G_AB.state_dict(), 'G_BA': model.G_BA.state_dict()}, final_path)
    print(f'\nGenerators saved → {final_path}')

    _plot_losses(args.out_dir)
