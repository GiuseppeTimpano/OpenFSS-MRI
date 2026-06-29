"""
Train SSBR (Self-Supervised Body Part Regression) on a directory of NIfTI volumes.

Images only, no labels. Volumes named image_<id>.nii.gz (label_<id>.nii.gz ignored).
Held-out ids are excluded from training so they can be used for validation / FSS test.

Example (server):
  PYTHONPATH=. python train_ssbr.py \
    --data_dir data/datasets/CHAOS/processed/T2 \
    --out ssbr_chaos_t2.pt \
    --heldout 1 2 3 10 13 20 21 \
    --steps 8000 --device cuda
"""
import argparse
import glob
import os
import random

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

from models.ssbr import SSBRNet, ssbr_order_equidist_loss


def load_vol(data_dir: str, sid: str) -> np.ndarray:
    """Load image_<sid>.nii.gz and per-volume z-score normalize (matches test.py)."""
    p = os.path.join(data_dir, f'image_{sid}.nii.gz')
    img = sitk.GetArrayFromImage(sitk.ReadImage(p)).astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)


def to_slice(vol_slice: np.ndarray, res: int) -> torch.Tensor:
    x = torch.from_numpy(vol_slice).unsqueeze(0).unsqueeze(0)        # [1,1,H,W]
    x = F.interpolate(x, size=(res, res), mode='bilinear', align_corners=False)
    return x.squeeze(0)                                             # [1,res,res]


def sample_sequence(vol: np.ndarray, m: int, res: int) -> torch.Tensor:
    """Equidistant ordered slices: random start, random gap, m slices -> [m,1,res,res]."""
    Z = vol.shape[0]
    max_g = max(1, (Z - 1) // (m - 1))
    g = random.randint(1, max_g)
    span = g * (m - 1)
    z0 = random.randint(0, Z - 1 - span)
    zs = [z0 + g * j for j in range(m)]
    return torch.stack([to_slice(vol[z], res) for z in zs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True, help='dir with image_<id>.nii.gz')
    ap.add_argument('--out',      required=True, help='output checkpoint path (.pt)')
    ap.add_argument('--heldout',  nargs='*', default=[], help='scan ids to exclude from training')
    ap.add_argument('--res',   type=int,   default=128)
    ap.add_argument('--m',     type=int,   default=10,   help='slices per sampled sequence')
    ap.add_argument('--alpha', type=float, default=1.0,  help='equidistance loss weight')
    ap.add_argument('--steps', type=int,   default=8000)
    ap.add_argument('--lr',    type=float, default=1e-3)
    ap.add_argument('--seed',  type=int,   default=0)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    heldout = set(args.heldout)

    ids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
           for p in glob.glob(os.path.join(args.data_dir, 'image_*.nii.gz'))]
    train_ids = [s for s in ids if s not in heldout]
    if not train_ids:
        raise ValueError(f'No training volumes in {args.data_dir} (all held out?)')
    print(f'train volumes ({len(train_ids)}): {sorted(train_ids, key=lambda x: int(x) if x.isdigit() else x)}')
    print(f'held-out     ({len(heldout)}): {sorted(heldout)}')

    # skip volumes too thin for a length-m sequence
    vols = {s: v for s in train_ids for v in [load_vol(args.data_dir, s)] if v.shape[0] >= args.m}
    train_ids = list(vols.keys())
    print(f'usable (Z>={args.m}): {len(train_ids)}')

    net = SSBRNet().to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for step in range(1, args.steps + 1):
        sid = random.choice(train_ids)
        seq = sample_sequence(vols[sid], args.m, args.res).to(device)   # [m,1,res,res]
        s = net(seq)
        loss, loss_order, loss_equi = ssbr_order_equidist_loss(s, args.alpha)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % 100 == 0 or step == 1:
            d = s.detach()
            mono = ((d[1:] - d[:-1]) > 0).float().mean().item()
            print(f'step {step:5d} | loss {loss.item():.4f} '
                  f'(order {loss_order.item():.4f}  equi {loss_equi.item():.4f}) | monotone {mono:.2f}')

    torch.save({'state_dict': net.state_dict(), 'res': args.res}, args.out)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
