# OpenFSS-MRI

Few-shot segmentation on CHAOS abdominal MRI (T1/T2). Work in progress.

---

## 1. Preprocessing

Convert raw CHAOS data to NIfTI, preprocess, and extract supervoxels:

```bash
python data/datasets/chaos.py
```

---

## 2. Training

Edit `configs/default.yaml` to set the data directory, fold, model, and settings, then run:

```bash
PYTHONPATH=. python scripts/prototype/train.py --config configs/default.yaml
```

---

## 3. Testing

```bash
PYTHONPATH=. python scripts/prototype/test.py --config configs/default.yaml --checkpoint path/to/ckpt.pth
```

For cross-domain testing (e.g. trained on T1, test on T2):

```bash
PYTHONPATH=. python scripts/prototype/test.py --config configs/default.yaml --checkpoint path/to/ckpt.pth \
    --target_data_dir data/datasets/CHAOS/processed/T2
```
