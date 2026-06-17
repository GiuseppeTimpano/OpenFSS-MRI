# OpenFSS-MRI

> **Work in progress** — PyTorch Lightning refactor of Q-Net and SSL-ALPNet for few-shot MRI segmentation.

Unified codebase for reproducible experiments on the [CHAOS](https://chaos.grand-challenge.org/) abdominal MRI dataset (T1/T2, 4 organs: Liver, RK, LK, Spleen).

---

## Models

| Model | Encoder | Prototype |
|-------|---------|-----------|
| `qnet` | ResNet101 dual-scale | GlobalPrototype + adaptive threshold |
| `alpnet` | ResNet101 (DeepLab) | GridConv+ / GridConv |

Both use the same training protocol: Q-Net neighbour (adjacent-slice) supervoxel sampling, SGD + MultiStepLR (lr=1e-3, γ=0.95 every 1000 steps).

---

## Setup

```bash
pip install -r requirements.txt
```

Preprocess CHAOS dataset (DICOM → NIfTI → supervoxels):
```bash
python data/datasets/chaos.py
```

---

## Training

```bash
python train.py --config configs/default.yaml
```

Key config options (`configs/default.yaml`):

```yaml
data:
  data_dir: data/datasets/CHAOS/processed/T1
  fold: 0          # 0–3
  exclude_label: null        # Setting 1 (null) or Setting 2 (e.g. [1,4])
model:
  name: qnet       # qnet | alpnet
train:
  bg_loss_weight: 0.1        # qnet: 0.1 | alpnet: 0.05
```

**Setting 1** — test organs visible in background during SSL training (`exclude_label: null`).  
**Setting 2** — slices containing test organs removed from SSL pool (`exclude_label: [1,4]` for Liver+Spleen, `[2,3]` for RK+LK).

---

## Testing

```bash
# same-domain
python test.py --config configs/default.yaml --checkpoint path/to/ckpt.pth

# cross-domain T1 → T2
python test.py --config configs/default.yaml --checkpoint path/to/ckpt.pth \
    --target_data_dir data/datasets/CHAOS/processed/T2
```

Outputs per-class Dice and IoU averaged over test patients.

---

## Experiment matrix

16 runs for a full comparison (4 folds × 2 models × 2 settings):

```bash
python train.py --config configs/qnet_s1_fold0.yaml    # Q-Net, Setting 1, fold 0
python train.py --config configs/qnet_s2_fold0.yaml    # Q-Net, Setting 2, fold 0
python train.py --config configs/alpnet_s1_fold0.yaml  # ALPNet, Setting 1, fold 0
# ...
```

---

## References

- **Q-Net**: *Self-supervised few-shot medical image segmentation with query-guided prototype* (2022)
- **SSL-ALPNet**: *Adaptive local-global prototype segmentation* (2021)
- **CHAOS**: Combined Healthy Abdominal Organ Segmentation challenge dataset
