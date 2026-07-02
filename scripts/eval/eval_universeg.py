"""
UniverSeg zero-shot eval -- in-context few-shot, deployable (NO oracle box). Frozen
pretrained model, direct challenger to the prototype baseline on the same
"support-based, no update" axis; reuses eval_common for the shared table.

Support protocol mirrors scripts/prototype/test.py (same pool/supp_idx/n_part),
except ALL n_part support slices are fed jointly to EVERY query slice (UniverSeg's
native usage), vs. test.py's one-support-slice-per-depth-chunk (ALPNet/QNet's
contract).

Normalization: percentile-clip + min-max [0,1] float (models/universeg_adapter.py
volume_to_unit_float) -- not the baseline's z-score, not MedSAM2's uint8+ImageNet.
"""
import argparse
import csv
import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from eval_common import Scores, aggregate_and_print
from models.universeg_adapter import UniverSegSegmenter, volume_to_unit_float


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_raw(data_dir: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    return img, lbl


def _support_indices(n_part: int, n_fg: int) -> np.ndarray:
    """Identical to test.py: n_part slice indices evenly spaced across n_fg FG slices."""
    if n_part == 1:
        pcts = [0.5]
    else:
        half_part     = 1.0 / (n_part * 2)
        part_interval = (1.0 - 1.0 / n_part) / (n_part - 1)
        pcts = [half_part + part_interval * i for i in range(n_part)]
    return (np.array(pcts) * n_fg).astype(int)


def _select_support_slices(supp_fg_idx: np.ndarray, n_part: int) -> np.ndarray:
    return supp_fg_idx[_support_indices(n_part, len(supp_fg_idx))]


def evaluate(cfg: dict, target_data_dir: str | None, fold: int | None,
             supp_idx: int, n_part: int, eval_labels: list[int] | None,
             device: str, save_dir: str | None, save_topk: int = 1) -> dict:
    data_cfg    = cfg['data']
    data_dir    = data_cfg['data_dir']
    n_folds     = data_cfg['n_folds']
    label_names = data_cfg['label_names']
    if eval_labels is None:
        eval_labels = list(range(1, len(label_names)))

    _, test_ids = get_fold_ids(data_dir, fold if fold is not None else data_cfg.get('fold', 0), n_folds)
    if not test_ids:
        raise ValueError(f'No test scans for fold {fold}')
    supp_sid = test_ids[supp_idx]

    query_data_dir = target_data_dir or data_dir
    if target_data_dir:
        paths = sorted(glob.glob(os.path.join(target_data_dir, 'image_*.nii.gz')))
        query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                      for p in paths]
    else:
        query_sids = [sid for sid in test_ids if sid != supp_sid]
    if not query_sids:
        raise ValueError(f'No query scans found in {query_data_dir}')

    print(f'UniverSeg in-context | support: {supp_sid} | queries={len(query_sids)} '
          f'| n_part={n_part} | eval_labels={eval_labels}')
    print(f'query dir: {query_data_dir}')

    supp_img, supp_lbl = _load_raw(data_dir, supp_sid)
    supp_img01 = volume_to_unit_float(supp_img)

    seg = UniverSegSegmenter(device=device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    class_dice: dict[str, float] = {}
    class_iou:  dict[str, float] = {}
    csv_rows: list[dict] = []

    for label_val in eval_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)
        print(f'\n== Class: {label_name} (label={label_val}) ==')

        supp_fg_mask = (supp_lbl == label_val).astype(np.uint8)
        supp_fg_idx  = np.where(supp_fg_mask.any(axis=(1, 2)))[0]
        if len(supp_fg_idx) == 0:
            print(f'  [SKIP] support {supp_sid} has no FG for {label_name}')
            continue

        sel_z = _select_support_slices(supp_fg_idx, n_part)
        s_imgs  = supp_img01[sel_z]
        s_masks = supp_fg_mask[sel_z]

        scores = Scores()
        scan_ids: list[str] = []
        # kept only for this class's loop, discarded once best/worst are written out
        cls_img: dict[str, np.ndarray] = {}
        cls_gt:  dict[str, np.ndarray] = {}
        cls_pred: dict[str, np.ndarray] = {}

        for qsid in query_sids:
            q_img, q_lbl = _load_raw(query_data_dir, qsid)
            q_fg = (q_lbl == label_val).astype(np.uint8)

            fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
            if len(fg_idx) == 0:
                print(f'  [SKIP] query {qsid} has no FG for {label_name}')
                continue

            z0, z1 = int(fg_idx.min()), int(fg_idx.max())
            q_vol01 = volume_to_unit_float(q_img)[z0:z1 + 1]
            seg_crop = seg.segment_volume(q_vol01, s_imgs, s_masks)   # [z1-z0+1,H,W]

            pred_full = np.zeros_like(q_fg)
            pred_full[z0:z1 + 1] = seg_crop

            pred_fg = torch.from_numpy(pred_full[fg_idx].astype(np.int64))
            gt_fg   = torch.from_numpy(q_fg[fg_idx].astype(np.int64))
            scores.record(pred_fg, gt_fg)
            scan_ids.append(qsid)
            print(f'  scan {qsid}: Dice={scores.patient_dice[-1]:.4f}  '
                  f'IoU={scores.patient_iou[-1]:.4f}')

            if save_dir and save_topk > 0:
                cls_img[qsid]  = q_img.astype(np.float32)
                cls_gt[qsid]   = q_fg
                cls_pred[qsid] = pred_full

        if scores.patient_dice:
            class_dice[label_name] = float(np.mean(scores.patient_dice))
            class_iou[label_name]  = float(np.mean(scores.patient_iou))
            print(f'  mean Dice={class_dice[label_name]:.4f}  '
                  f'mean IoU={class_iou[label_name]:.4f}')

            for sid, d, i in zip(scan_ids, scores.patient_dice, scores.patient_iou):
                csv_rows.append({'class': label_name, 'label': label_val,
                                  'scan': sid, 'dice': d, 'iou': i})

            if save_dir and save_topk > 0 and scan_ids:
                order = sorted(range(len(scan_ids)), key=lambda k: scores.patient_dice[k])
                worst_idx = set(order[:save_topk])
                best_idx  = set(order[-save_topk:])
                for k in worst_idx | best_idx:
                    sid = scan_ids[k]
                    tag = 'best' if k in best_idx else 'worst'
                    d = scores.patient_dice[k]
                    base = f'{label_name}_{tag}_{sid}_dice{d:.3f}'
                    sitk.WriteImage(sitk.GetImageFromArray(cls_img[sid]),
                                    os.path.join(save_dir, f'{base}_image.nii.gz'), True)
                    sitk.WriteImage(sitk.GetImageFromArray(cls_gt[sid]),
                                    os.path.join(save_dir, f'{base}_gt.nii.gz'), True)
                    sitk.WriteImage(sitk.GetImageFromArray(cls_pred[sid]),
                                    os.path.join(save_dir, f'{base}_pred.nii.gz'), True)

    if save_dir and csv_rows:
        csv_path = os.path.join(save_dir, 'scores.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'label', 'scan', 'dice', 'iou'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f'\nPer-scan scores written to {csv_path}')

    results = aggregate_and_print(class_dice, class_iou)

    if save_dir and results:
        summary_path = os.path.join(save_dir, 'summary.csv')
        with open(summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['class', 'dice', 'iou'])
            writer.writeheader()
            for name, vals in results.items():
                writer.writerow({'class': name, 'dice': vals['dice'], 'iou': vals['iou']})
        print(f'Summary written to {summary_path}')

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str, default='configs/resnet.yaml')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='query dir (processed image_*/label_*); default = config data_dir')
    parser.add_argument('--fold',            type=int, default=None)
    parser.add_argument('--supp_idx',        type=int, default=0)
    parser.add_argument('--n_part',          type=int, default=3,
                        help='number of support FG slices (same default as test.py)')
    parser.add_argument('--test_label',      type=int, nargs='+', default=None)
    parser.add_argument('--save_dir',        type=str, default=None,
                        help='where to write scores.csv/summary.csv and best/worst volumes')
    parser.add_argument('--save_topk',       type=int, default=1,
                        help='per class, save nii.gz for N best + N worst scans (0 = CSV only, no volumes)')
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    evaluate(
        cfg_file,
        target_data_dir = args.target_data_dir,
        fold            = args.fold,
        supp_idx        = args.supp_idx,
        n_part          = args.n_part,
        eval_labels     = args.test_label,
        device          = args.device,
        save_dir        = args.save_dir,
        save_topk       = args.save_topk,
    )
