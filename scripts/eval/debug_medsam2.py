"""
Debug tool for the MedSAM2 eval on MRI_muscle. Two subcommands:

  triage <scores.csv>          per-class table + failing scans, worst first.
  vis --box_source oracle      box = query GT bbox, no matching -> isolates SAM2 itself.
  vis --box_source support     box from support matching (prompt_mode=support_bbox).

Filename encodes dice, and boxiou (2D box-vs-GT IoU on the prompt slice) for support:
boxiou~0 = mislocation (matching problem); boxiou high + low dice = SAM2 problem.
Comparing oracle vs support on the same query separates the two.

  PYTHONPATH=. python3 scripts/eval/debug_medsam2.py vis --box_source oracle ...
Wrappers: scripts/eval/run_debug.sh <experiment>
"""
import argparse
import csv
import glob
import os
import random
from collections import defaultdict
from statistics import mean, median

import numpy as np

# ============================== triage (CSV only, no model) ==============================


def _load_scores(path: str) -> list[dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['dice'] = float(r['dice'])
        r['iou'] = float(r['iou'])
    return rows


def cmd_triage(args) -> None:
    """Reads the per-scan scores.csv written by eval_medsam2.py (class,label,scan,dice,iou)."""
    rows = _load_scores(args.scores_csv)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r['class']].append(r)

    stats = []
    for cls, rs in by_class.items():
        d = [r['dice'] for r in rs]
        stats.append((cls, len(d), mean(d), median(d), min(d), max(d)))
    stats.sort(key=lambda s: s[2])  # worst class first

    print(f'\n=== Per-class (n={len(rows)} scans, {len(by_class)} classes) ===')
    print(f'{"class":<8}{"n":>4}{"mean":>9}{"median":>9}{"min":>9}{"max":>9}')
    for cls, n, mn, md, lo, hi in stats:
        print(f'{cls:<8}{n:>4}{mn:>9.4f}{md:>9.4f}{lo:>9.4f}{hi:>9.4f}')
    overall = [r['dice'] for r in rows]
    print(f'{"ALL":<8}{len(overall):>4}{mean(overall):>9.4f}'
          f'{median(overall):>9.4f}{min(overall):>9.4f}{max(overall):>9.4f}')

    print(f'\n=== Scans below threshold (Dice < {args.thr}) ===')
    any_fail = False
    for cls, *_ in stats:
        fails = sorted((r for r in by_class[cls] if r['dice'] < args.thr),
                       key=lambda r: r['dice'])
        if fails:
            any_fail = True
            ids = ', '.join(f'{r["scan"]}({r["dice"]:.3f})' for r in fails)
            print(f'  {cls:<8} {len(fails):>2}/{len(by_class[cls]):<2}  {ids}')
    if not any_fail:
        print('  none.')

    print('\n=== 10 worst scans overall ===')
    for r in sorted(rows, key=lambda r: r['dice'])[:10]:
        print(f'  {r["class"]:<8} {r["scan"]:<20} Dice={r["dice"]:.4f} IoU={r["iou"]:.4f}')


# ============================== vis (loads MedSAM2) ==============================
# same defaults as support_prompt_for_query_dense_bodymasked_bbox
THR_HI, THR_LO = 0.7, 0.3
BODY_THRESH, BODY_MIN_PX = 10.0, 50
SCORE_THRESH, MARGIN_PX = 0.0, 0.0


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def _gt_bbox(mask2d: np.ndarray):
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


def _predict_box_support(seg, supp_frame_u8, supp_mask2d, query_frames):
    """Mirrors support_prompt_for_query_dense_bodymasked_bbox but keeps the intermediates
    (does not modify support_prompt.py). Returns the max-score frame:
    (frame_idx, box, score, pos_map, neg_map, frame_u8)."""
    from models.support_prompt import (body_mask2d, extract_support_vectors_bodymasked,
                                       dense_similarity_maps, bbox_from_similarity_blob)
    supp_feat = seg.embed_frame(supp_frame_u8)
    supp_body = body_mask2d(supp_frame_u8, BODY_THRESH, BODY_MIN_PX)
    Pos_n, Neg_n = extract_support_vectors_bodymasked(supp_feat, supp_mask2d, supp_body,
                                                      THR_HI, THR_LO)
    best = None
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
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(map2d.astype(np.float32))[None, None]
    return F.interpolate(t, size=hw, mode='bilinear', align_corners=False)[0, 0].numpy()


def _draw_box(ax, box, lw=2.5):
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor='cyan', linewidth=lw))


def _render(out_png, panels: list, titles: list):
    """panels[i] draws on axis i."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 6))
    for ax, draw, title in zip(np.atleast_1d(axes), panels, titles):
        draw(ax)
        ax.set_title(title, fontsize=10, pad=8)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=110, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)


def cmd_vis(args) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import torch
    import yaml
    from models.medsam2_adapter import MedSAM2Segmenter, volume_to_uint8
    from models.support_prompt import key_slice
    from eval_medsam2 import _read_nii, _load_raw, _build_boxes

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    label_names = cfg['data']['label_names']
    test_labels = args.test_labels or list(range(1, len(label_names)))

    paths = sorted(glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
    # P* = pathological patients, excluded from query AND support pool (as in evaluate())
    all_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                for p in paths if not os.path.basename(p).startswith('image_P')]
    keep = set(all_sids if not args.only else [s for s in all_sids if s in set(args.only)])
    if not keep:
        raise ValueError('No query scans found (check --only / --target_data_dir)')

    os.makedirs(args.out_dir, exist_ok=True)
    seg = MedSAM2Segmenter(args.medsam2_ckpt, args.sam2_cfg, device=device)

    for label_val in test_labels:
        label_name = label_names[label_val] if label_val < len(label_names) else str(label_val)

        support_fg_idx = {}
        if args.box_source == 'support':
            for sid in all_sids:
                lbl = _read_nii(os.path.join(args.target_data_dir, f'label_{sid}.nii.gz'))
                if np.where((lbl == label_val).any(axis=(1, 2)))[0].size:
                    support_fg_idx[sid] = True
        rng = random.Random(args.seed + label_val)

        # iterate over ALL queries (not just --only) to consume the rng in evaluate()'s
        # sequence -> identical support/query pairings
        for qsid in all_sids:
            q_img, q_lbl = _load_raw(args.target_data_dir, qsid)
            q_fg = (q_lbl == label_val).astype(np.uint8)
            fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]

            if args.box_source == 'support':
                pool = [s for s in support_fg_idx if s != qsid]
                if len(fg_idx) == 0 or not pool:
                    continue
                rng_supp = rng.choice(pool)   # always consume the rng
            elif len(fg_idx) == 0:
                continue
            if qsid not in keep:
                continue

            z0, z1 = int(fg_idx.min()), int(fg_idx.max())
            vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]
            fg_crop = q_fg[z0:z1 + 1]

            if args.box_source == 'oracle':
                mode = 'key' if args.query_slice == 'key' else 'perslice'
                boxes = _build_boxes(fg_crop, fg_idx, z0, mode, args.margin)
                seg_crop = seg.segment_volume(vol_u8, boxes, refine_iters=args.refine_iters)
                pred_full = np.zeros_like(q_fg)
                pred_full[z0:z1 + 1] = seg_crop
                d = _dice(pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool))

                frame_idx = key_slice(q_fg) - z0   # key slice is always in boxes if mode=key
                if frame_idx not in boxes:
                    frame_idx = next(iter(boxes))
                box = boxes[frame_idx]
                frame_u8, gt2d = vol_u8[frame_idx], fg_crop[frame_idx].astype(bool)
                pred2d = seg_crop[frame_idx].astype(bool)

                def p0(ax, f=frame_u8, g=gt2d, b=box):
                    ax.imshow(f, cmap='gray'); ax.contour(g, colors='yellow', linewidths=1.5)
                    _draw_box(ax, b)

                def p1(ax, f=frame_u8, g=gt2d, pr=pred2d, b=box):
                    ax.imshow(f, cmap='gray'); ax.contour(g, colors='yellow', linewidths=1.5)
                    ax.contour(pr, colors='red', linewidths=1.5); _draw_box(ax, b, 2.0)

                out = os.path.join(args.out_dir, f'{label_name}_{qsid}_oracle_{mode}_dice{d:.3f}.png')
                _render(out, [p0, p1],
                        [f'ORACLE box (GT) + GT — {mode}',
                         f'{qsid} [{label_name}]  Dice(vol)={d:.3f}'])
                print(f'[{label_name}] {qsid}: oracle Dice={d:.4f} -> {os.path.basename(out)}')
                continue

            # --- box_source == support ---
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

                frame_idx, box, conf, pos_map, neg_map, q_frame_u8 = _predict_box_support(
                    seg, supp_frame_u8, supp_mask2d, query_frames)
                seg_crop = seg.segment_volume(vol_u8, {frame_idx: np.asarray(box, dtype=np.float32)},
                                              refine_iters=args.refine_iters)
                pred_full = np.zeros_like(q_fg)
                pred_full[z0:z1 + 1] = seg_crop
                d = _dice(pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool))

                gt2d = fg_crop[frame_idx].astype(bool)
                pred2d = seg_crop[frame_idx].astype(bool)
                boxiou = _box_iou(tuple(box), _gt_bbox(gt2d))
                score_up = _upsample(pos_map - neg_map, q_frame_u8.shape)

                def s0(ax, f=supp_frame_u8, m=supp_mask2d):
                    ax.imshow(f, cmap='gray'); ax.contour(m, colors='yellow', linewidths=1.5)

                def s1(ax, f=q_frame_u8, s=score_up, g=gt2d, b=box):
                    ax.imshow(f, cmap='gray'); ax.imshow(s, cmap='jet', alpha=0.45)
                    ax.contour(g, colors='yellow', linewidths=1.5); _draw_box(ax, b)

                def s2(ax, f=q_frame_u8, g=gt2d, pr=pred2d, b=box):
                    ax.imshow(f, cmap='gray'); ax.contour(g, colors='yellow', linewidths=1.5)
                    ax.contour(pr, colors='red', linewidths=1.5); _draw_box(ax, b, 2.0)

                tag = f'_supp{supp_sid}' if args.all_supports else ''
                out = os.path.join(args.out_dir,
                                   f'{label_name}_{qsid}{tag}_dice{d:.3f}_boxiou{boxiou:.2f}.png')
                _render(out, [s0, s1, s2],
                        [f'support {supp_sid} (z={supp_z})',
                         f'similarity + BOX  conf={conf:.3f}  boxiou={boxiou:.2f}',
                         f'{qsid} z={z0 + frame_idx}  Dice(vol)={d:.3f}'])
                print(f'[{label_name}] {qsid}: support={supp_sid} Dice={d:.4f} '
                      f'boxiou={boxiou:.3f} conf={conf:.3f} -> {os.path.basename(out)}')


# ============================== CLI ==============================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('triage', help='per-class table + failing scans from a scores.csv')
    t.add_argument('scores_csv')
    t.add_argument('--thr', type=float, default=0.40, help='Dice below this = failed scan')
    t.set_defaults(func=cmd_triage)

    v = sub.add_parser('vis', help='debug PNGs (box + segmentation)')
    v.add_argument('--box_source', choices=['oracle', 'support'], required=True,
                   help='oracle = box from query GT (isolates SAM2); '
                        'support = box from support matching (isolates the matching)')
    v.add_argument('--config', required=True)
    v.add_argument('--medsam2_ckpt', required=True)
    v.add_argument('--sam2_cfg', required=True)
    v.add_argument('--target_data_dir', required=True)
    v.add_argument('--test_labels', type=int, nargs='+', default=None,
                   help='default: every non-BG label in the config')
    v.add_argument('--query_slice', choices=['auto', 'key'], default='auto',
                   help='support: auto = similarity picks the slice, key = max-area. '
                        'oracle: auto => per-slice boxes (upper bound), key => a single box')
    v.add_argument('--margin', type=int, default=0, help='box_source=oracle only')
    v.add_argument('--refine_iters', type=int, default=1)
    v.add_argument('--seed', type=int, default=42, help='must match the eval to reproduce pairings')
    v.add_argument('--only', nargs='+', default=None, help='limit to these query sids')
    v.add_argument('--all_supports', action='store_true',
                   help='box_source=support only: draw EVERY candidate support (variance)')
    v.add_argument('--device', default=None)
    v.add_argument('--out_dir', required=True)
    v.set_defaults(func=cmd_vis)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
