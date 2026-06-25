"""
CycleGAN training for medical MRI domain adaptation.

Experiments:
  T2→T1 (cross-modality):
    python -m cyclegan.train_cyclegan \
        --src_dir data/datasets/CHAOS/processed/T2 \
        --tgt_dir data/datasets/CHAOS/processed/T1 \
        --out_dir cyclegan/checkpoints/T2_to_T1

  AMOS→CHAOS (cross-dataset):
    python -m cyclegan.train_cyclegan \
        --src_dir data/datasets/AMOS/processed/T2 \
        --tgt_dir data/datasets/CHAOS/processed/T2 \
        --out_dir cyclegan/checkpoints/AMOS_to_CHAOS
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cyclegan.models import UNetGenerator2D, NLayerDiscriminator2D
from cyclegan.dataset import UnpairedNIfTIDataset


class _ReplayBuffer:
    """Stores up to max_size previous fake images; randomly replays old samples."""

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


def _set_grad(nets: list[nn.Module], requires_grad: bool):
    for net in nets:
        for p in net.parameters():
            p.requires_grad = requires_grad


def _save_samples(real_A, real_B, fake_B, epoch: int, out_dir: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        imgs   = [real_A[0, 0], real_B[0, 0], fake_B[0, 0]]
        titles = ['real_A (src)', 'real_B (tgt)', 'fake_B = G_AB(A)']
        for ax, img, title in zip(axes, imgs, titles):
            ax.imshow(img.detach().cpu().numpy(), cmap='gray', vmin=0, vmax=1)
            ax.set_title(title, fontsize=8)
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'sample_ep{epoch:03d}.png'), dpi=100)
        plt.close()
    except Exception as e:
        print(f'  [sample save skipped: {e}]')


def train(args):
    device = torch.device(args.device)

    G_AB = UNetGenerator2D().to(device)
    G_BA = UNetGenerator2D().to(device)
    D_A  = NLayerDiscriminator2D().to(device)
    D_B  = NLayerDiscriminator2D().to(device)

    opt_G   = torch.optim.Adam(
        list(G_AB.parameters()) + list(G_BA.parameters()),
        lr=args.lr, betas=(0.5, 0.999),
    )
    opt_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

    decay_start = args.epochs // 2

    def lr_lambda(epoch: int) -> float:
        if epoch < decay_start:
            return 1.0
        return 1.0 - (epoch - decay_start) / max(1, args.epochs - decay_start)

    sch_G   = torch.optim.lr_scheduler.LambdaLR(opt_G,   lr_lambda)
    sch_D_A = torch.optim.lr_scheduler.LambdaLR(opt_D_A, lr_lambda)
    sch_D_B = torch.optim.lr_scheduler.LambdaLR(opt_D_B, lr_lambda)

    crit_adv = nn.BCEWithLogitsLoss()
    crit_cyc = nn.L1Loss()
    crit_id  = nn.L1Loss()

    dataset = UnpairedNIfTIDataset(args.src_dir, args.tgt_dir)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    buf_A = _ReplayBuffer()
    buf_B = _ReplayBuffer()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Training {args.epochs} epochs | device={args.device} | '
          f'batch={args.batch_size} | λ_cyc={args.lambda_cyc} | λ_id={args.lambda_id}')

    for epoch in range(args.epochs):
        G_AB.train(); G_BA.train(); D_A.train(); D_B.train()

        sum_G = 0.0
        sum_D = 0.0

        for real_A, real_B in loader:
            real_A = real_A.to(device)   # [B, 1, H, W] ∈ [0,1]
            real_B = real_B.to(device)

            #  Generator update 
            _set_grad([D_A, D_B], False)
            opt_G.zero_grad()

            fake_B  = G_AB(real_A)
            fake_A  = G_BA(real_B)
            rec_A   = G_BA(fake_B)
            rec_B   = G_AB(fake_A)

            # Adversarial (generator wants D to predict real)
            loss_adv = (
                crit_adv(D_B(fake_B), torch.ones_like(D_B(fake_B))) +
                crit_adv(D_A(fake_A), torch.ones_like(D_A(fake_A)))
            )
            # Cycle consistency
            loss_cyc = (crit_cyc(rec_A, real_A) + crit_cyc(rec_B, real_B)) * args.lambda_cyc
            # Identity
            loss_id  = (crit_id(G_BA(real_A), real_A) + crit_id(G_AB(real_B), real_B)) * args.lambda_id

            loss_G = loss_adv + loss_cyc + loss_id
            loss_G.backward()
            opt_G.step()

            #  Discriminator update 
            _set_grad([D_A, D_B], True)

            # D_A: real_A vs fake_A (from buffer)
            opt_D_A.zero_grad()
            fake_A_buf   = buf_A.push_and_pop(fake_A.detach())
            pred_real_A  = D_A(real_A)
            pred_fake_A  = D_A(fake_A_buf)
            loss_D_A = (
                crit_adv(pred_real_A, torch.ones_like(pred_real_A) * 0.9) +
                crit_adv(pred_fake_A, torch.zeros_like(pred_fake_A))
            ) * 0.5
            loss_D_A.backward()
            opt_D_A.step()

            # D_B: real_B vs fake_B (from buffer)
            opt_D_B.zero_grad()
            fake_B_buf   = buf_B.push_and_pop(fake_B.detach())
            pred_real_B  = D_B(real_B)
            pred_fake_B  = D_B(fake_B_buf)
            loss_D_B = (
                crit_adv(pred_real_B, torch.ones_like(pred_real_B) * 0.9) +
                crit_adv(pred_fake_B, torch.zeros_like(pred_fake_B))
            ) * 0.5
            loss_D_B.backward()
            opt_D_B.step()

            sum_G += loss_G.item()
            sum_D += loss_D_A.item() + loss_D_B.item()

        sch_G.step(); sch_D_A.step(); sch_D_B.step()

        n = len(loader)
        print(f'Epoch [{epoch + 1:03d}/{args.epochs}]  '
              f'G={sum_G / n:.4f}  D={sum_D / n:.4f}  '
              f'lr={sch_G.get_last_lr()[0]:.6f}')

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = os.path.join(args.out_dir, f'cyclegan_ep{epoch + 1:03d}.pth')
            torch.save({
                'epoch':   epoch + 1,
                'G_AB':    G_AB.state_dict(),
                'G_BA':    G_BA.state_dict(),
                'D_A':     D_A.state_dict(),
                'D_B':     D_B.state_dict(),
                'opt_G':   opt_G.state_dict(),
                'opt_D_A': opt_D_A.state_dict(),
                'opt_D_B': opt_D_B.state_dict(),
                'args':    vars(args),
            }, ckpt)
            print(f'  → {ckpt}')
            _save_samples(real_A, real_B, fake_B, epoch + 1, args.out_dir)


#  Entry point

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir',    required=True,  help='Source domain NIfTI dir')
    parser.add_argument('--tgt_dir',    required=True,  help='Target domain NIfTI dir')
    parser.add_argument('--out_dir',    default='cyclegan/checkpoints')
    parser.add_argument('--epochs',     type=int,   default=200)
    parser.add_argument('--batch_size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=2e-4)
    parser.add_argument('--lambda_cyc', type=float, default=10.0)
    parser.add_argument('--lambda_id',  type=float, default=5.0)
    parser.add_argument('--save_every', type=int,   default=10)
    parser.add_argument('--workers',    type=int,   default=4)
    parser.add_argument('--device',     default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)
