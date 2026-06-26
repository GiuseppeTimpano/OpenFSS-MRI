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
import lightning as L
from lightning.pytorch.callbacks import RichProgressBar, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

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
    def __init__(self, lr=2e-4, lambda_cyc=10.0, lambda_id=5.0, epochs=200):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.G_AB = UNetGenerator2D()
        self.G_BA = UNetGenerator2D()
        self.D_A  = NLayerDiscriminator2D()
        self.D_B  = NLayerDiscriminator2D()

        self.crit_adv = nn.BCEWithLogitsLoss()
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
        real_A, real_B = batch
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
        loss_G = loss_adv + loss_cyc + loss_id

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
            "loss_cyc": loss_cyc, "loss_id": loss_id,
            "loss_D_A": loss_D_A, "loss_D_B": loss_D_B,
        }, on_step=False, on_epoch=True, prog_bar=True)

        if (self.current_epoch + 1) % 10 == 0 and batch_idx == 0:
            self._save_samples(real_A, real_B, fake_B)

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
                ax.imshow(img.detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
                ax.set_title(title, fontsize=8); ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f'sample_ep{self.current_epoch + 1:03d}.png'), dpi=100)
            plt.close()
        except Exception as e:
            print(f'[sample save skipped: {e}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',    required=True)
    parser.add_argument('--tgt_dir',    required=True)
    parser.add_argument('--out_dir',    default='cyclegan/runs')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--lambda_cyc', type=float, default=10.0)
    parser.add_argument('--lambda_id',  type=float, default=5.0)
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--save_ckpt',  action='store_true',
                        help='Save full Lightning checkpoint (default: only generators saved)')
    parser.add_argument('--device',     default='auto')
    args = parser.parse_args()

    dataset = UnpairedNIfTIDataset(args.src_dir, args.tgt_dir)
    loader  = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )

    model = CycleGAN(lr=args.lr, lambda_cyc=args.lambda_cyc,
                     lambda_id=args.lambda_id, epochs=args.epochs)

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
