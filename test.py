"""
Volumetric few-shot test script.
Protocol identical to original Q-Net test.py / SSL-ALPNet eval (CHAOS dataset):
  - test fold scan IDs from get_fold_ids(fold)
  - 1 support scan (supp_idx, default 0), N = n_part support slices evenly spaced in FG
  - for each query scan: only FG slices, split into n_part chunks; chunk i uses support slice i
  - metric: per-patient 3D Dice + IoU, mean per class over all test patients
"""

import argparse
import json
import os

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from models.fewshot import FewShotConfig, QNetFewShot, ALPNetFewShot


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_scan(data_dir: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    """Load image + label for scan sid; normalize image per-volume."""
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img, lbl


def _support_indices(n_part: int, n_fg: int) -> np.ndarray:
    """
    Select n_part slice indices evenly spaced across n_fg foreground slices.
    Identical to Q-Net TestDataset.get_support_index (called with N=n_part).
    """
    if n_part == 1:
        pcts = [0.5]
    else:
        half_part     = 1.0 / (n_part * 2)
        part_interval = (1.0 - 1.0 / n_part) / (n_part - 1)
        pcts = [half_part + part_interval * i for i in range(n_part)]
    return (np.array(pcts) * n_fg).astype(int)


class Scores:
    """Accumulates per-patient 3D Dice and IoU (same as Q-Net utils.Scores)."""

    def __init__(self):
        self.patient_dice: list[float] = []
        self.patient_iou:  list[float] = []

    def record(self, pred: torch.Tensor, label: torch.Tensor):
        tp = ((label == 1) & (pred == 1)).sum().float()
        fp = ((label == 0) & (pred == 1)).sum().float()
        fn = ((label == 1) & (pred == 0)).sum().float()
        dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
        iou  = (tp / (tp + fp + fn + 1e-8)).item()
        self.patient_dice.append(dice)
        self.patient_iou.append(iou)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',     type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='path to Lightning .ckpt or raw model .pth')
    parser.add_argument('--fold',       type=int, default=None,
                        help='override data.fold from config')
    parser.add_argument('--supp_idx',   type=int, default=None,
                        help='which test scan to use as support (default: from config)')
    parser.add_argument('--n_part',     type=int, default=None,
                        help='number of support slices / query chunks (default: from config)')
    parser.add_argument('--test_label', type=int, nargs='+', default=None,
                        help='organ label indices to evaluate (default: all non-BG)')
    parser.add_argument('--save_dir',   type=str, default=None,
                        help='if set, save binary prediction nii.gz files here')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='query scans directory for cross-domain test (e.g. T2 when trained on T1)')
    parser.add_argument('--device',     type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    data_cfg  = cfg_file['data']
    model_name = cfg_file['model']['name']
    test_cfg  = cfg_file.get('test', {})

    fold      = args.fold     if args.fold     is not None else data_cfg['fold']
    n_folds   = data_cfg['n_folds']
    data_dir  = data_cfg['data_dir']
    n_shot    = data_cfg['n_shot']
    supp_idx  = args.supp_idx if args.supp_idx is not None else test_cfg.get('supp_idx', 0)
    n_part    = args.n_part   if args.n_part   is not None else test_cfg.get('n_part', 3)

    label_names: list[str] = data_cfg['label_names']  # [BG, LIVER, RK, LK, SPLEEN]
    name_to_idx = {name: idx for idx, name in enumerate(label_names)}

    # classes to evaluate (skip BG=0)
    if args.test_label is not None:
        eval_labels = args.test_label
    elif test_cfg.get('test_label'):
        eval_labels = test_cfg['test_label']
    else:
        eval_labels = [i for i in range(1, len(label_names))]

    query_data_dir = args.target_data_dir if args.target_data_dir else data_dir

    domain_cfg  = cfg_file.get('domain', {})
    domain_map  = None
    if domain_cfg.get('domain_map'):
        with open(domain_cfg['domain_map']) as f:
            domain_map = json.load(f)
    source_domain = domain_cfg.get('source_domain')
    target_domain = domain_cfg.get('target_domain')

    # build model
    device = torch.device(args.device)
    cfg    = FewShotConfig(encoder_type=model_name, n_shot=n_shot)

    bg_loss_weight = cfg_file.get('train', {}).get('bg_loss_weight', 0.1)
    if model_name == 'qnet':
        model = QNetFewShot(cfg, bg_loss_weight=bg_loss_weight)
    else:
        model = ALPNetFewShot(cfg, bg_loss_weight=bg_loss_weight)

    # load from Lightning .ckpt (strips '_model.' prefix) or raw .pth
    raw = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in raw:
        state = {k.removeprefix('_model.'): v
                 for k, v in raw['state_dict'].items()
                 if k.startswith('_model.')}
    else:
        state = raw
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # fold split 
    _, test_ids = get_fold_ids(data_dir, fold, n_folds)
    if not test_ids:
        raise ValueError(f'No test scans for fold {fold}')

    # apply cross-domain filter if requested
    if domain_map and source_domain and target_domain:
        supp_pool  = [sid for sid in test_ids if domain_map.get(sid) == source_domain]
        query_pool = [sid for sid in test_ids if domain_map.get(sid) == target_domain]
        if not supp_pool or not query_pool:
            raise ValueError(f'No scans for domains {source_domain}/{target_domain} in test fold')
    else:
        supp_pool  = test_ids
        query_pool = test_ids

    supp_sid   = supp_pool[supp_idx]
    # when querying a different directory, supp_sid and query scans are distinct files
    if args.target_data_dir:
        query_sids = list(query_pool)
    else:
        query_sids = [sid for sid in query_pool if sid != supp_sid]

    print(f'Fold {fold}  |  support scan: {supp_sid}  |  query scans: {query_sids}')
    print(f'n_part={n_part}, eval_labels={eval_labels}')

    # load support volume once
    supp_img, supp_lbl = _load_scan(data_dir, supp_sid)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # per-class evaluation
    class_dice: dict[str, float] = {}
    class_iou:  dict[str, float] = {}

    for label_val in eval_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)
        print(f'\n== Class: {label_name} (label={label_val}) ==')

        # support: select n_part slices evenly spaced in FG
        supp_fg_mask  = (supp_lbl == label_val).astype(np.float32)
        supp_fg_idx   = np.where(supp_fg_mask.any(axis=(1, 2)))[0]
        if len(supp_fg_idx) == 0:
            print(f'  [SKIP] support scan {supp_sid} has no foreground for {label_name}')
            continue

        sel_idx  = _support_indices(n_part, len(supp_fg_idx))
        sel_z    = supp_fg_idx[sel_idx]     # absolute z indices into support volume

        # build support tensors: list of n_part (1-shot) entries
        # each entry: supp_imgs [1,1,H,W], supp_masks [1,1,H,W]
        sup_imgs_list  = []
        sup_masks_list = []
        for z in sel_z:
            si = torch.from_numpy(supp_img[z]).to(device).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
            sm = torch.from_numpy(supp_fg_mask[z]).to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            sup_imgs_list.append(si)
            sup_masks_list.append(sm)

        scores = Scores()

        for qsid in query_sids:
            q_img, q_lbl = _load_scan(query_data_dir, qsid)
            q_fg_mask    = (q_lbl == label_val).astype(np.float32)

            # only evaluate on FG slices (original protocol)
            fg_idx = np.where(q_fg_mask.any(axis=(1, 2)))[0]
            if len(fg_idx) == 0:
                print(f'  [SKIP] query scan {qsid} has no foreground for {label_name}')
                continue

            q_img_fg     = q_img[fg_idx]      # [C_q, H, W]
            q_fg_mask_fg = q_fg_mask[fg_idx]  # [C_q, H, W]
            C_q          = len(fg_idx)

            # split into n_part chunks (same as original)
            chunk_bounds = np.linspace(0, C_q, n_part + 1).astype(int)

            H, W           = q_img_fg.shape[1], q_img_fg.shape[2]
            query_pred_vol = torch.zeros(C_q, H, W, dtype=torch.long)

            with torch.no_grad():
                for chunk_i in range(n_part):
                    s_img  = sup_imgs_list[chunk_i]
                    s_mask = sup_masks_list[chunk_i]
                    a, b   = chunk_bounds[chunk_i], chunk_bounds[chunk_i + 1]
                    for j in range(a, b):
                        qi = torch.from_numpy(q_img_fg[j]).to(device).unsqueeze(0)  # [1,H,W]
                        pred = model(s_img, s_mask, qi)                              # [1,2,H,W]
                        query_pred_vol[j] = pred.argmax(dim=1).cpu().squeeze(0)

            q_label_vol = torch.from_numpy(q_fg_mask_fg).long()
            scores.record(query_pred_vol, q_label_vol)

            dice_val = scores.patient_dice[-1]
            iou_val  = scores.patient_iou[-1]
            print(f'  scan {qsid}: Dice={dice_val:.4f}  IoU={iou_val:.4f}')

            if args.save_dir:
                pred_np = query_pred_vol.numpy().astype(np.uint8)
                itk_img = sitk.GetImageFromArray(pred_np)
                out_path = os.path.join(args.save_dir, f'pred_{qsid}_{label_name}.nii.gz')
                sitk.WriteImage(itk_img, out_path, True)

        if scores.patient_dice:
            class_dice[label_name] = float(np.mean(scores.patient_dice))
            class_iou[label_name]  = float(np.mean(scores.patient_iou))
            print(f'  mean Dice={class_dice[label_name]:.4f}  mean IoU={class_iou[label_name]:.4f}')

    print('\n===== Final results =====')
    for name in class_dice:
        print(f'  {name}: Dice={class_dice[name]:.4f}  IoU={class_iou[name]:.4f}')
    if class_dice:
        all_dice = list(class_dice.values())
        all_iou  = list(class_iou.values())
        print(f'  MEAN:  Dice={np.mean(all_dice):.4f}  IoU={np.mean(all_iou):.4f}')


if __name__ == '__main__':
    main()
