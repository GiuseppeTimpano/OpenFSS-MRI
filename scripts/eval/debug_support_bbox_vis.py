"""
Visual debug per prompt_mode=support_bbox (models/support_prompt.py's dense +
body-mask + similarity-BLOB-BBOX path). Companion a scripts/eval/eval_medsam2.py
--prompt_mode support_bbox: per ogni query salva un PNG a 3 pannelli per VEDERE
dove il box predetto finisce rispetto al muscolo vero.

Pannelli:
  1. support: frame + GT mask (giallo) -- da cosa si estraggono le support-vector.
  2. query prompt-frame: heatmap similarita' (pos_map-neg_map) + BOX predetto (ciano)
     + GT contour (giallo). Mostra SE il blob di similarita' cade sul muscolo giusto.
  3. query prompt-frame: BOX (ciano) + GT (giallo) + segmentazione finale (rosso).
     Mostra se, dato il box, la maschera propagata e' corretta.

Diagnostica numerica stampata/nel filename:
  box_gt_iou = IoU(box predetto, bbox del GT sulla prompt-slice).
    box_gt_iou basso  -> Regime A: box sul tessuto SBAGLIATO (problema di matching).
    box_gt_iou alto + Dice basso -> box giusto ma maschera/propagazione fallisce.

NON modifica support_prompt.py: replica esattamente il corpo di
support_prompt_for_query_dense_bodymasked_bbox (stessi default: thr_hi=0.7,
thr_lo=0.3, body_thresh=10, body_min_px=50, score_thresh=0.0, margin_px=0.0),
tenendo pero' gli intermedi (pos_map/neg_map/box) per il disegno.

rng seed + sequenza identici alla branch support_bbox di evaluate(), quindi il
support scelto (e il Dice) coincidono 1:1 con una eval con stesso --seed --query_slice.
Con --all_supports invece ignora l'rng e disegna OGNI support candidato (per vedere
la varianza: quanto il risultato dipende da CHI e' il support).
"""
import argparse
import glob
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import yaml

from models.medsam2_adapter import MedSAM2Segmenter, volume_to_uint8
from models.support_prompt import (key_slice, body_mask2d, extract_support_vectors_bodymasked,
                                    dense_similarity_maps, bbox_from_similarity_blob)

# default identici a support_prompt_for_query_dense_bodymasked_bbox
THR_HI, THR_LO = 0.7, 0.3
BODY_THRESH, BODY_MIN_PX = 10.0, 50
SCORE_THRESH, MARGIN_PX = 0.0, 0.0


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _load_raw(data_dir: str, sid: str):
    img = _read_nii(os.path.join(data_dir, f'image_{sid}.nii.gz')).astype(np.float32)
    lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz')).astype(np.int32)
    return img, lbl


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def _gt_bbox(mask2d: np.ndarray):
    """(x0,y0,x1,y1) del GT sulla slice, o None se vuota."""
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _box_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return float(inter / ua) if ua > 0 else 0.0


def _predict_box(seg, supp_frame_u8, supp_mask2d, query_frames):
    """Replica support_prompt_for_query_dense_bodymasked_bbox tenendo gli intermedi.
    Ritorna (frame_idx, box, score, pos_map, neg_map, frame_u8) del frame a max score."""
    supp_feat = seg.embed_frame(supp_frame_u8)
    supp_body = body_mask2d(supp_frame_u8, BODY_THRESH, BODY_MIN_PX)
    Pos_n, Neg_n = extract_support_vectors_bodymasked(supp_feat, supp_mask2d, supp_body,
                                                      THR_HI, THR_LO)
    best = None  # (score, fidx, box, pos_map, neg_map, frame_u8)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, BODY_THRESH, BODY_MIN_PX)
        box = bbox_from_similarity_blob(pos_map, neg_map, q_body, frame_u8.shape,
                                        SCORE_THRESH, MARGIN_PX)
        score = float((pos_map - neg_map).max())
        if best is None or score > best[0]:
            best = (score, fidx, box, pos_map, neg_map, frame_u8)
    return best[1], best[2], best[0], best[3], best[4], best[5]


def _upsample(map2d: np.ndarray, hw) -> np.ndarray:
    t = torch.from_numpy(map2d.astype(np.float32))[None, None]
    return F.interpolate(t, size=hw, mode='bilinear', align_corners=False)[0, 0].numpy()


def _render(out_png, supp_frame_u8, supp_mask2d, supp_sid, supp_z,
            q_frame_u8, prompted_z, box, gt2d, pred2d, score_map, conf, d, box_gt_iou, qsid):
    H, W = q_frame_u8.shape
    score_up = _upsample(score_map, (H, W))
    x0, y0, x1, y1 = box

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(supp_frame_u8, cmap='gray')
    ax[0].contour(supp_mask2d, colors='yellow', linewidths=1.5)
    ax[0].set_title(f'support {supp_sid} (z={supp_z})', fontsize=10, pad=8)
    ax[0].axis('off')

    ax[1].imshow(q_frame_u8, cmap='gray')
    ax[1].imshow(score_up, cmap='jet', alpha=0.45)
    ax[1].contour(gt2d, colors='yellow', linewidths=1.5)
    ax[1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                              edgecolor='cyan', linewidth=2.5))
    ax[1].set_title(f'similarity + BOX  conf={conf:.3f}  box_gt_iou={box_gt_iou:.2f}',
                    fontsize=10, pad=8)
    ax[1].axis('off')

    ax[2].imshow(q_frame_u8, cmap='gray')
    ax[2].contour(gt2d, colors='yellow', linewidths=1.5)
    ax[2].contour(pred2d, colors='red', linewidths=1.5)
    ax[2].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                              edgecolor='cyan', linewidth=2.0))
    ax[2].set_title(f'{qsid} z={prompted_z}  Dice(vol)={d:.3f}', fontsize=10, pad=8)
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--medsam2_ckpt', required=True)
    ap.add_argument('--sam2_cfg', required=True)
    ap.add_argument('--target_data_dir', required=True)
    ap.add_argument('--test_label', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42,
                    help='deve combaciare con la eval per riprodurre gli stessi support')
    ap.add_argument('--query_slice', choices=['auto', 'key'], default='auto',
                    help='auto = la similarita\' sceglie la slice; key = slice max-area (proxy operatore)')
    ap.add_argument('--refine_iters', type=int, default=1)
    ap.add_argument('--only', nargs='+', default=None,
                    help='limita a questi query sid (es. HV002_1_stack2), default tutti')
    ap.add_argument('--all_supports', action='store_true',
                    help='per ogni query disegna OGNI support candidato (varianza), non solo quello rng')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    label_names = cfg['data']['label_names']
    label_val = args.test_label
    label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)

    paths = sorted(glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
    query_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '') for p in paths]
    query_sids = [s for s in query_sids if not s.startswith('P')]
    if args.only:
        query_sids = [s for s in query_sids if s in set(args.only)]
    if not query_sids:
        raise ValueError('Nessuna query trovata (controlla --only / --target_data_dir)')

    os.makedirs(args.out_dir, exist_ok=True)
    seg = MedSAM2Segmenter(args.medsam2_ckpt, args.sam2_cfg, device=args.device)

    rng = random.Random(args.seed + label_val)
    support_fg_idx = {}
    for sid in [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                for p in paths if not os.path.basename(p).startswith('image_P')]:
        lbl = _read_nii(os.path.join(args.target_data_dir, f'label_{sid}.nii.gz'))
        idx = np.where((lbl == label_val).any(axis=(1, 2)))[0]
        if len(idx):
            support_fg_idx[sid] = idx

    # NB: l'rng va consumato su TUTTE le query (come la eval) per far combaciare i
    # pairing anche quando --only ne seleziona un sottoinsieme.
    all_queries = [s for s in [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                               for p in paths] if not s.startswith('P')]
    keep = set(query_sids)

    for qsid in all_queries:
        q_img, q_lbl = _load_raw(args.target_data_dir, qsid)
        q_fg = (q_lbl == label_val).astype(np.uint8)
        fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
        pool = [s for s in support_fg_idx if s != qsid]
        if len(fg_idx) == 0 or not pool:
            if qsid in keep:
                print(f'[SKIP] {qsid}: no FG / no support')
            continue
        rng_supp = rng.choice(pool)   # consuma rng SEMPRE (allinea i pairing)
        if qsid not in keep:
            continue

        z0, z1 = int(fg_idx.min()), int(fg_idx.max())
        vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]

        if args.query_slice == 'key':
            zc = key_slice(q_fg)
            query_frames = [(zc - z0, vol_u8[zc - z0])]
        else:
            query_frames = [(int(z) - z0, vol_u8[int(z) - z0]) for z in fg_idx]

        supports = pool if args.all_supports else [rng_supp]
        for supp_sid in supports:
            supp_img, supp_lbl = _load_raw(args.target_data_dir, supp_sid)
            supp_fg = (supp_lbl == label_val).astype(np.uint8)
            supp_z = key_slice(supp_fg)
            supp_frame_u8 = volume_to_uint8(supp_img)[supp_z]
            supp_mask2d = supp_fg[supp_z].astype(bool)

            frame_idx, box, score, pos_map, neg_map, q_frame_u8 = _predict_box(
                seg, supp_frame_u8, supp_mask2d, query_frames)
            seg_crop = seg.segment_volume(vol_u8, {frame_idx: np.asarray(box, dtype=np.float32)},
                                          refine_iters=args.refine_iters)

            pred_full = np.zeros_like(q_fg)
            pred_full[z0:z1 + 1] = seg_crop
            d = _dice(pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool))

            prompted_z = z0 + frame_idx
            gt2d = q_fg[prompted_z].astype(bool)
            pred2d = seg_crop[frame_idx].astype(bool)
            box_gt_iou = _box_iou(tuple(box), _gt_bbox(gt2d))

            tag = f'_supp{supp_sid}' if args.all_supports else ''
            out = os.path.join(args.out_dir,
                               f'{label_name}_{qsid}{tag}_dice{d:.3f}_boxiou{box_gt_iou:.2f}.png')
            _render(out, supp_frame_u8, supp_mask2d, supp_sid, supp_z,
                    q_frame_u8, prompted_z, box, gt2d, pred2d, pos_map - neg_map, score,
                    d, box_gt_iou, qsid)
            print(f'{qsid}: support={supp_sid} Dice={d:.4f} box_gt_iou={box_gt_iou:.3f} '
                  f'conf={score:.3f} -> {os.path.basename(out)}')


if __name__ == '__main__':
    main()
