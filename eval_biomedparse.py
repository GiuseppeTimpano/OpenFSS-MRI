"""
BiomedParse zero-shot evaluation (P1-style) — volume-level, text-prompted, no box/
support set. Third paradigm alongside eval_medsam2.py (box+propagation) and
eval_universeg.py (support set), see models/biomedparse_adapter.py docstring.

Reuses the identical scorer (eval_common.Scores) and per-organ + MEAN aggregation
so its numbers sit in the same table as the other adapters / prototype baseline.

Protocol (per organ, per query volume):
  - crop to the FG depth range (same scoring stack as the other eval_*.py scripts).
  - one text prompt for the whole volume (models.biomedparse_adapter.PROMPT_TEMPLATES),
    one forward pass, no oracle box, no support scan — this is BiomedParse's native
    deployable usage (fully zero-shot given the text prompt).

Image normalization: uint8 [0,255] percentile windowing (same convention as MedSAM2's
volume_to_uint8), NOT the z-score of the baseline nor UniverSeg's [0,1] float — a
model requirement, see models/biomedparse_adapter.py.

IMPORTANT (contamination): BiomedParse v2 is trained on AMOS + TotalSegmentator-MRI
(see HANDOFF.md). Only run this on datasets confirmed clean for it (CirrMRI).
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
import torch
import yaml

from data.dataloader.dataset import get_fold_ids
from eval_common import Scores, aggregate_and_print
from models.biomedparse_adapter import BiomedParseSegmenter, PROMPT_TEMPLATES, volume_to_uint8


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_raw(data_dir: str, sid: str) -> tuple[np.ndarray, np.ndarray]:
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    return img, lbl


def evaluate(cfg: dict, checkpoint: str | None, target_data_dir: str | None,
             fold: int | None, eval_labels: list[int] | None,
             device: str, save_dir: str | None, limit: int | None = None) -> dict:
    data_cfg    = cfg['data']
    data_dir    = data_cfg['data_dir']
    n_folds     = data_cfg['n_folds']
    label_names = data_cfg['label_names']
    if eval_labels is None:
        eval_labels = list(range(1, len(label_names)))

    query_data_dir = target_data_dir or data_dir
    if target_data_dir:
        paths = sorted(glob.glob(os.path.join(target_data_dir, 'image_*.nii.gz')))
        query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                      for p in paths]
    else:
        _, query_sids = get_fold_ids(data_dir, fold if fold is not None else 0, n_folds)
    if not query_sids:
        raise ValueError(f'No query scans found in {query_data_dir}')
    if limit:
        query_sids = query_sids[:limit]

    print(f'BiomedParse zero-shot (text prompt) | queries={len(query_sids)} '
          f'| eval_labels={eval_labels}')
    print(f'query dir: {query_data_dir}')

    seg = BiomedParseSegmenter(checkpoint, device=device)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    multi_img:  dict[str, np.ndarray] = {}
    multi_gt:   dict[str, np.ndarray] = {}
    multi_pred: dict[str, np.ndarray] = {}

    class_dice: dict[str, float] = {}
    class_iou:  dict[str, float] = {}

    for label_val in eval_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)
        prompt = PROMPT_TEMPLATES.get(label_name)
        if prompt is None:
            print(f'\n== Class: {label_name} (label={label_val}) == [SKIP] no prompt template')
            continue
        print(f'\n== Class: {label_name} (label={label_val}) == prompt="{prompt}"')
        scores = Scores()

        for qsid in query_sids:
            q_img, q_lbl = _load_raw(query_data_dir, qsid)
            q_fg = (q_lbl == label_val).astype(np.uint8)

            if save_dir and qsid not in multi_pred:
                multi_img[qsid]  = q_img.astype(np.float32)
                multi_gt[qsid]   = q_lbl.astype(np.uint8)
                multi_pred[qsid] = np.zeros_like(q_lbl, dtype=np.uint8)

            fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
            if len(fg_idx) == 0:
                print(f'  [SKIP] query {qsid} has no FG for {label_name}')
                continue

            z0, z1 = int(fg_idx.min()), int(fg_idx.max())
            vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]          # window full vol, then crop

            seg_crop = seg.segment_volume(vol_u8, prompt)        # [z1-z0+1,H,W]
            pred_full = np.zeros_like(q_fg)
            pred_full[z0:z1 + 1] = seg_crop

            pred_fg = torch.from_numpy(pred_full[fg_idx].astype(np.int64))
            gt_fg   = torch.from_numpy(q_fg[fg_idx].astype(np.int64))
            scores.record(pred_fg, gt_fg)
            print(f'  scan {qsid}: Dice={scores.patient_dice[-1]:.4f}  '
                  f'IoU={scores.patient_iou[-1]:.4f}')

            if save_dir:
                for z in range(z0, z1 + 1):
                    multi_pred[qsid][z][pred_full[z] == 1] = label_val

        if scores.patient_dice:
            class_dice[label_name] = float(np.mean(scores.patient_dice))
            class_iou[label_name]  = float(np.mean(scores.patient_iou))
            print(f'  mean Dice={class_dice[label_name]:.4f}  '
                  f'mean IoU={class_iou[label_name]:.4f}')

    if save_dir:
        for qsid in multi_pred:
            sitk.WriteImage(sitk.GetImageFromArray(multi_img[qsid]),
                            os.path.join(save_dir, f'{qsid}_image.nii.gz'), True)
            sitk.WriteImage(sitk.GetImageFromArray(multi_gt[qsid]),
                            os.path.join(save_dir, f'{qsid}_gt.nii.gz'), True)
            sitk.WriteImage(sitk.GetImageFromArray(multi_pred[qsid]),
                            os.path.join(save_dir, f'{qsid}_pred.nii.gz'), True)

    return aggregate_and_print(class_dice, class_iou)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',          type=str, default='configs/resnet.yaml')
    parser.add_argument('--biomedparse_ckpt', type=str, default=None,
                        help='local biomedparse_v2.ckpt; default = download from HF hub')
    parser.add_argument('--target_data_dir', type=str, default=None,
                        help='query dir (processed image_*/label_*); default = config data_dir')
    parser.add_argument('--fold',            type=int, default=None,
                        help='only used when --target_data_dir is not given')
    parser.add_argument('--test_label',      type=int, nargs='+', default=None)
    parser.add_argument('--limit',           type=int, default=None,
                        help='cap number of query scans (smoke test)')
    parser.add_argument('--save_dir',        type=str, default=None)
    parser.add_argument('--device',          type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_file = yaml.safe_load(f)

    evaluate(
        cfg_file,
        checkpoint      = args.biomedparse_ckpt,
        target_data_dir = args.target_data_dir,
        fold            = args.fold,
        eval_labels     = args.test_label,
        device          = args.device,
        save_dir        = args.save_dir,
        limit           = args.limit,
    )
