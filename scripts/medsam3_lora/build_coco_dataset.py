"""
NIfTI volumes -> COCO segmentation dataset for
third_party/MedSAM3/train_sam3_lora_native.py's COCOSegmentDataset: writes
<out_dir>/{train,valid}/_annotations.coco.json + PNGs, one 2D slice per image,
RLE segmentation, category name = lowercased text query at train time.

Reuses models.medsam3_adapter.volume_to_uint8 + PROMPT_TEMPLATES so LoRA training
sees the same normalization/prompt convention as zero-shot eval. Patient-level
split (data.dataloader.dataset.get_fold_ids) -- no slice leakage across splits.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/medsam3_lora/build_coco_dataset.py \\
      --data_dir data/datasets/CHAOS/processed/T1 \\
      --out_dir  data/datasets/CHAOS_coco/T1 \\
      --labels 1 2 3 4 --label_names BG LIVER RK LK SPLEEN
"""
import argparse
import glob
import json
import os

import numpy as np
import pycocotools.mask as mask_utils
import SimpleITK as sitk
from PIL import Image as PILImage

from data.dataloader.dataset import get_fold_ids
from models.medsam3_adapter import PROMPT_TEMPLATES, volume_to_uint8


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _mask_to_ann(mask: np.ndarray, min_area: int) -> dict | None:
    """Binary [H,W] -> COCO annotation fields, or None if too small/empty."""
    ys, xs = np.where(mask)
    if len(ys) < min_area:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]  # COCO xywh

    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('ascii')  # bytes -> JSON-safe str

    return {
        'bbox': bbox,
        'area': int(mask.sum()),
        'segmentation': rle,
        'iscrowd': 0,
    }


def build_split(data_dir: str, sids: list[str], labels: list[int],
                 label_names: list[str], out_split_dir: str, min_area: int) -> None:
    os.makedirs(out_split_dir, exist_ok=True)

    categories = [{'id': lv, 'name': PROMPT_TEMPLATES[label_names[lv]]}
                  for lv in labels if label_names[lv] in PROMPT_TEMPLATES]
    cat_ids = {c['id'] for c in categories}

    images, annotations = [], []
    img_id = 0
    ann_id = 0

    for sid in sids:
        img_path = os.path.join(data_dir, f'image_{sid}.nii.gz')
        lbl_path = os.path.join(data_dir, f'label_{sid}.nii.gz')
        if not (os.path.exists(img_path) and os.path.exists(lbl_path)):
            print(f'  [SKIP] missing image/label for scan {sid}')
            continue

        img = _read_nii(img_path).astype(np.float32)
        lbl = _read_nii(lbl_path).astype(np.int32)
        vol_u8 = volume_to_uint8(img)  # same windowing as eval_medsam3.py

        target_mask = np.isin(lbl, list(cat_ids))
        fg_idx = np.where(target_mask.any(axis=(1, 2)))[0]

        for z in fg_idx:
            slice_anns = []
            for lv in cat_ids:
                obj_mask = (lbl[z] == lv)
                ann = _mask_to_ann(obj_mask, min_area)
                if ann is None:
                    continue
                ann['id'] = ann_id
                ann['image_id'] = img_id
                ann['category_id'] = lv
                slice_anns.append(ann)
                ann_id += 1
            if not slice_anns:
                continue  # all target labels below min_area on this slice

            h, w = vol_u8.shape[1:]
            file_name = f'{sid}_z{z:03d}.png'
            PILImage.fromarray(vol_u8[z]).convert('RGB').save(
                os.path.join(out_split_dir, file_name))
            images.append({'id': img_id, 'file_name': file_name, 'width': w, 'height': h})
            annotations.extend(slice_anns)
            img_id += 1

        print(f'  scan {sid}: {len(fg_idx)} FG slices scanned')

    coco = {'images': images, 'annotations': annotations, 'categories': categories}
    ann_path = os.path.join(out_split_dir, '_annotations.coco.json')
    with open(ann_path, 'w') as f:
        json.dump(coco, f)
    print(f'{out_split_dir}: {len(images)} images, {len(annotations)} annotations '
          f'-> {ann_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                         help='source processed dir with image_*/label_*.nii.gz')
    parser.add_argument('--out_dir', type=str, required=True,
                         help='writes out_dir/train and out_dir/valid')
    parser.add_argument('--labels', type=int, nargs='+', default=[1, 2, 3, 4],
                         help='label ids to export as COCO categories/annotations')
    parser.add_argument('--label_names', type=str, nargs='+',
                         default=['BG', 'LIVER', 'RK', 'LK', 'SPLEEN'],
                         help='index i = label id i (must include index 0 for BG); '
                              'names must be keys in models.medsam3_adapter.PROMPT_TEMPLATES')
    parser.add_argument('--fold', type=int, default=0,
                         help='get_fold_ids chunk used as the VALID split')
    parser.add_argument('--n_folds', type=int, default=5,
                         help='patient-level split granularity, e.g. 5 -> ~20%% valid')
    parser.add_argument('--min_area', type=int, default=50,
                         help='skip a per-slice mask with fewer than this many FG pixels')
    args = parser.parse_args()

    missing = [args.label_names[lv] for lv in args.labels
               if args.label_names[lv] not in PROMPT_TEMPLATES]
    if missing:
        raise ValueError(f'No PROMPT_TEMPLATES entry for label names {missing} -- '
                          f'add them to models/medsam3_adapter.py PROMPT_TEMPLATES first')

    train_ids, valid_ids = get_fold_ids(args.data_dir, args.fold, args.n_folds)
    print(f'{len(train_ids)} train scans / {len(valid_ids)} valid scans '
          f'(fold={args.fold}/{args.n_folds}, patient-level split)')
    print(f'valid scan ids: {sorted(valid_ids, key=int)}')

    print('\n== train split ==')
    build_split(args.data_dir, train_ids, args.labels, args.label_names,
                os.path.join(args.out_dir, 'train'), args.min_area)
    print('\n== valid split ==')
    build_split(args.data_dir, valid_ids, args.labels, args.label_names,
                os.path.join(args.out_dir, 'valid'), args.min_area)


if __name__ == '__main__':
    main()
