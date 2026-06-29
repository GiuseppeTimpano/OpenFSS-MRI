"""
Volumetric few-shot test script.
Protocol identical to original Q-Net test.py / SSL-ALPNet eval (CHAOS dataset):
  - test fold scan IDs from get_fold_ids(fold)
  - 1 support scan (supp_idx, default 0), N = n_part support slices evenly spaced in FG
  - for each query scan: only FG slices, split into n_part chunks; chunk i uses support slice i
  - metric: per-patient 3D Dice + IoU, mean per class over all test patients

Support-slice selection is pluggable via --support_select:
  percentile (default) reproduces the original protocol above; ssbr replaces the
  chunk->support mapping with body-part-score nearest-neighbour matching (see
  models/ssbr.py), letting each query slice pick its closest support slice without
  using the query label.
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
from models.ssbr import load_ssbr, score_volume


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


def test_from_cfg(
    cfg: dict,
    checkpoint: str,
    target_data_dir: str | None = None,
    device_str: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str | None = None,
    support_select: str = 'percentile',
    ssbr_ckpt: str | None = None,
) -> dict:
    """
    Run volumetric test. Returns:
      {'LIVER': {'dice': x, 'iou': y}, ..., 'MEAN': {'dice': x, 'iou': y}}

    support_select:
      'percentile' (default, original protocol) — pick n_part support slices evenly
        spaced in the support FG extent; split the query FG slices into n_part
        contiguous chunks (by relative depth) and pair chunk i with support slice i.
      'ssbr' — score every support and query slice with a trained SSBR body-part
        regressor; for each query FG slice pick the support FG slice with the nearest
        body-part score. Deployable: selection uses the query *image* only (no query
        label), the query label is used solely to evaluate Dice.
    """
    data_cfg   = cfg['data']
    model_cfg  = cfg['model']
    model_name = model_cfg['name']
    test_cfg   = cfg.get('test', {})

    fold         = data_cfg['fold']
    n_folds      = data_cfg['n_folds']
    data_dir     = data_cfg['data_dir']
    n_shot       = data_cfg['n_shot']
    supp_idx     = test_cfg.get('supp_idx', 0)
    n_part       = test_cfg.get('n_part', 3)
    label_names  = data_cfg['label_names']

    if test_cfg.get('test_label'):
        eval_labels = test_cfg['test_label']
    else:
        eval_labels = list(range(1, len(label_names)))

    query_data_dir = target_data_dir or data_dir

    domain_cfg    = cfg.get('domain', {})
    domain_map    = None
    if domain_cfg.get('domain_map'):
        with open(domain_cfg['domain_map']) as f:
            domain_map = json.load(f)
    source_domain = domain_cfg.get('source_domain')
    target_domain = domain_cfg.get('target_domain')

    device = torch.device(device_str)
    fcfg = FewShotConfig(
        encoder_type = model_name,
        n_shot       = n_shot,
    )

    bg_loss_weight = cfg.get('train', {}).get('bg_loss_weight', 0.1)
    model = QNetFewShot(fcfg, bg_loss_weight=bg_loss_weight) \
            if model_name == 'qnet' \
            else ALPNetFewShot(fcfg, bg_loss_weight=bg_loss_weight)

    raw = torch.load(checkpoint, map_location='cpu')
    if 'state_dict' in raw:
        state = {k.removeprefix('_model.'): v
                 for k, v in raw['state_dict'].items()
                 if k.startswith('_model.')}
    else:
        state = raw
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    if support_select not in ('percentile', 'ssbr'):
        raise ValueError(f"support_select must be 'percentile' or 'ssbr', got {support_select}")
    ssbr_net = ssbr_res = None
    if support_select == 'ssbr':
        if not ssbr_ckpt:
            raise ValueError("support_select='ssbr' requires --ssbr_ckpt")
        ssbr_net, ssbr_res = load_ssbr(ssbr_ckpt, device)
        print(f'SSBR support selection: {ssbr_ckpt} (res={ssbr_res})')

    _, test_ids = get_fold_ids(data_dir, fold, n_folds)
    if not test_ids:
        raise ValueError(f'No test scans for fold {fold}')

    if domain_map and source_domain and target_domain:
        supp_pool  = [sid for sid in test_ids if domain_map.get(sid) == source_domain]
        query_pool = [sid for sid in test_ids if domain_map.get(sid) == target_domain]
    else:
        supp_pool  = test_ids
        query_pool = test_ids

    supp_sid   = supp_pool[supp_idx]
    if target_data_dir:
        import glob as _glob
        _paths = sorted(_glob.glob(os.path.join(target_data_dir, 'image_*.nii.gz')))
        query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '') for p in _paths]
    else:
        query_sids = [sid for sid in query_pool if sid != supp_sid]

    print(f'Fold {fold}  |  support: {supp_sid}  |  queries: {query_sids}')
    print(f'n_part={n_part}, eval_labels={eval_labels}')

    supp_img, supp_lbl = _load_scan(data_dir, supp_sid)
    supp_ssbr = score_volume(ssbr_net, supp_img, ssbr_res, device) \
                if support_select == 'ssbr' else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    class_dice: dict[str, float] = {}
    class_iou:  dict[str, float] = {}

    for label_val in eval_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)
        print(f'\n== Class: {label_name} (label={label_val}) ==')

        supp_fg_mask = (supp_lbl == label_val).astype(np.float32)
        supp_fg_idx  = np.where(supp_fg_mask.any(axis=(1, 2)))[0]
        if len(supp_fg_idx) == 0:
            print(f'  [SKIP] support {supp_sid} has no FG for {label_name}')
            continue

        if support_select == 'ssbr':
            sel_z = supp_fg_idx                      # full FG pool as candidates
            supp_sel_score = supp_ssbr[sel_z]        # body-part score per candidate
        else:
            sel_z = supp_fg_idx[_support_indices(n_part, len(supp_fg_idx))]
            supp_sel_score = None

        sup_imgs_list  = []
        sup_masks_list = []
        for z in sel_z:
            si = torch.from_numpy(supp_img[z]).to(device).unsqueeze(0).unsqueeze(0)
            sm = torch.from_numpy(supp_fg_mask[z]).to(device).unsqueeze(0).unsqueeze(0)
            sup_imgs_list.append(si)
            sup_masks_list.append(sm)

        scores = Scores()

        for qsid in query_sids:
            q_img, q_lbl = _load_scan(query_data_dir, qsid)
            q_fg_mask    = (q_lbl == label_val).astype(np.float32)

            fg_idx = np.where(q_fg_mask.any(axis=(1, 2)))[0]
            if len(fg_idx) == 0:
                print(f'  [SKIP] query {qsid} has no FG for {label_name}')
                continue

            q_img_fg     = q_img[fg_idx]
            q_fg_mask_fg = q_fg_mask[fg_idx]
            C_q          = len(fg_idx)

            # assign[j] = index into sup_imgs_list used to segment query FG slice j
            if support_select == 'ssbr':
                q_fg_score = score_volume(ssbr_net, q_img, ssbr_res, device)[fg_idx]
                assign = np.array([int(np.argmin(np.abs(qs - supp_sel_score)))
                                   for qs in q_fg_score])
            else:
                chunk_bounds = np.linspace(0, C_q, n_part + 1).astype(int)
                assign = np.empty(C_q, dtype=int)
                for chunk_i in range(n_part):
                    a, b = chunk_bounds[chunk_i], chunk_bounds[chunk_i + 1]
                    assign[a:b] = chunk_i

            H, W           = q_img_fg.shape[1], q_img_fg.shape[2]
            query_pred_vol = torch.zeros(C_q, H, W, dtype=torch.long)

            with torch.no_grad():
                for j in range(C_q):
                    s_img  = sup_imgs_list[assign[j]]
                    s_mask = sup_masks_list[assign[j]]
                    qi     = torch.from_numpy(q_img_fg[j]).to(device).unsqueeze(0)
                    pred   = model(s_img, s_mask, qi)
                    query_pred_vol[j] = pred.argmax(dim=1).cpu().squeeze(0)

            q_label_vol = torch.from_numpy(q_fg_mask_fg).long()
            scores.record(query_pred_vol, q_label_vol)

            dice_val = scores.patient_dice[-1]
            iou_val  = scores.patient_iou[-1]
            print(f'  scan {qsid}: Dice={dice_val:.4f}  IoU={iou_val:.4f}')

            if save_dir:
                pred_np  = query_pred_vol.numpy().astype(np.uint8)
                itk_img  = sitk.GetImageFromArray(pred_np)
                out_path = os.path.join(save_dir, f'pred_{qsid}_{label_name}.nii.gz')
                sitk.WriteImage(itk_img, out_path, True)

        if scores.patient_dice:
            class_dice[label_name] = float(np.mean(scores.patient_dice))
            class_iou[label_name]  = float(np.mean(scores.patient_iou))
            print(f'  mean Dice={class_dice[label_name]:.4f}  mean IoU={class_iou[label_name]:.4f}')

    print('\n===== Final results =====')
    results = {}
    for name in class_dice:
        results[name] = {'dice': class_dice[name], 'iou': class_iou[name]}
        print(f'  {name}: Dice={class_dice[name]:.4f}  IoU={class_iou[name]:.4f}')
    if class_dice:
        mean_d = float(np.mean(list(class_dice.values())))
        mean_i = float(np.mean(list(class_iou.values())))
        results['MEAN'] = {'dice': mean_d, 'iou': mean_i}
        print(f'  MEAN:  Dice={mean_d:.4f}  IoU={mean_i:.4f}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str, default='configs/resnet.yaml')
    parser.add_argument('--checkpoint',      type=str, required=True)
    parser.add_argument('--fold',            type=int, default=None)
    parser.add_argument('--supp_idx',        type=int, default=None)
    parser.add_argument('--n_part',          type=int, default=None)
    parser.add_argument('--test_label',      type=int, nargs='+', default=None)
    parser.add_argument('--save_dir',        type=str, default=None)
    parser.add_argument('--target_data_dir', type=str, default=None)
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--support_select',  type=str, default='percentile',
                        choices=['percentile', 'ssbr'],
                        help='support-slice selection: percentile (original) or ssbr')
    parser.add_argument('--ssbr_ckpt',       type=str, default=None,
                        help='trained SSBR checkpoint (required for --support_select ssbr)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    if args.fold is not None:
        cfg_file['data']['fold'] = args.fold
    if args.supp_idx is not None:
        cfg_file.setdefault('test', {})['supp_idx'] = args.supp_idx
    if args.n_part is not None:
        cfg_file.setdefault('test', {})['n_part'] = args.n_part
    if args.test_label is not None:
        cfg_file.setdefault('test', {})['test_label'] = args.test_label

    test_from_cfg(
        cfg_file,
        checkpoint      = args.checkpoint,
        target_data_dir = args.target_data_dir,
        device_str      = args.device,
        save_dir        = args.save_dir,
        support_select  = args.support_select,
        ssbr_ckpt       = args.ssbr_ckpt,
    )
