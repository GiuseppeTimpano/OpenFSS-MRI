"""
Debug tool for the MedSAM2 eval on MRI_muscle. Two subcommands:

  triage <run>                 Dice/IoU per class + failing scans, worst first. <run> is a
                               scores.csv or the dir holding it -- works on ANY experiment,
                               eval_medsam2.py or vis, since both write the same schema.
  vis --box_source oracle      box = query GT bbox, no matching -> isolates SAM2 itself.
  vis --box_source support     box from support matching (prompt_mode=support_bbox).
  mcvis                        B2: box from cross-class competition, frozen slice. Same
                               seed/pairing/slice as `vis --box_source support
                               --query_slice key`, so boxiou is directly comparable.

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


CSV_FIELDS = ['class', 'label', 'scan', 'dice', 'iou']
Z_FIELDS = ['class', 'scan', 'z', 'dice', 'z_prompt']


def _load_scores(path: str) -> list[dict]:
    if os.path.isdir(path):
        path = os.path.join(path, 'scores.csv')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['dice'] = float(r['dice'])
        r['iou'] = float(r['iou'])
        if r.get('boxiou') not in (None, ''):
            r['boxiou'] = float(r['boxiou'])
    return rows


BANDS = [(0.0, 'prompt'), (0.25, '0-25%'), (0.5, '25-50%'),
         (0.75, '50-75%'), (1.0, '75-100%')]


def _print_propagation(path: str) -> None:
    """Mean Dice by normalized z-distance from the prompt slice, if dice_by_z.csv is there.
    Flat = propagation holds; falling = the mask is lost away from the prompt."""
    d = path if os.path.isdir(path) else os.path.dirname(path)
    z_path = os.path.join(d, 'dice_by_z.csv')
    if not os.path.exists(z_path):
        return
    with open(z_path, newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or int(rows[0]['z_prompt']) < 0:
        print('\n=== Propagation === per-slice prompts, nothing propagates.')
        return

    span: dict[tuple, int] = {}   # (class,scan) -> max |z - z_prompt|
    for r in rows:
        k = (r['class'], r['scan'])
        span[k] = max(span.get(k, 0), abs(int(r['z']) - int(r['z_prompt'])))

    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        k = (r['class'], r['scan'])
        far = abs(int(r['z']) - int(r['z_prompt'])) / max(1, span[k])
        band = next(name for hi, name in BANDS if far <= hi)
        buckets[(r['class'], band)].append(float(r['dice']))

    classes = sorted({r['class'] for r in rows},
                     key=lambda c: mean(buckets[(c, '75-100%')] or [0]))
    names = [n for _, n in BANDS]
    print('\n=== Propagation: mean Dice by distance from the prompt slice ===')
    print(f'{"class":<8}' + ''.join(f'{n:>10}' for n in names))
    for c in classes:
        cells = ''.join(f'{mean(buckets[(c, n)]):>10.4f}' if buckets[(c, n)] else f'{"-":>10}'
                        for n in names)
        print(f'{c:<8}{cells}')


def cmd_triage(args) -> None:
    """Reads a per-scan scores.csv (class,label,scan,dice,iou[,boxiou]) written by
    eval_medsam2.py or by `vis`. Accepts the csv path or its directory."""
    rows = _load_scores(args.scores_csv)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r['class']].append(r)

    has_box = any('boxiou' in r for r in rows)  # support runs only

    def _bi(rs):  # mean boxiou, or None
        v = [r['boxiou'] for r in rs if 'boxiou' in r]
        return mean(v) if v else None

    stats = []
    for cls, rs in by_class.items():
        d = [r['dice'] for r in rs]
        stats.append((cls, len(d), mean(d), median(d), min(d), max(d),
                      mean(r['iou'] for r in rs), _bi(rs)))
    stats.sort(key=lambda s: s[2])  # worst class first

    box_hdr = f'{"boxiou":>9}' if has_box else ''
    print(f'\n=== Per-class (n={len(rows)} scans, {len(by_class)} classes) ===')
    print(f'{"class":<8}{"n":>4}{"mean":>9}{"median":>9}{"min":>9}{"max":>9}'
          f'{"iou":>9}{box_hdr}')
    for cls, n, mn, md, lo, hi, iu, bi in stats:
        line = f'{cls:<8}{n:>4}{mn:>9.4f}{md:>9.4f}{lo:>9.4f}{hi:>9.4f}{iu:>9.4f}'
        print(line + (f'{bi:>9.4f}' if has_box and bi is not None else ''))
    overall = [r['dice'] for r in rows]
    all_bi = _bi(rows)
    line = (f'{"ALL":<8}{len(overall):>4}{mean(overall):>9.4f}{median(overall):>9.4f}'
            f'{min(overall):>9.4f}{max(overall):>9.4f}'
            f'{mean(r["iou"] for r in rows):>9.4f}')
    print(line + (f'{all_bi:>9.4f}' if has_box and all_bi is not None else ''))

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
        b = f' boxiou={r["boxiou"]:.3f}' if 'boxiou' in r else ''
        print(f'  {r["class"]:<8} {r["scan"]:<20} Dice={r["dice"]:.4f} IoU={r["iou"]:.4f}{b}')

    _print_propagation(args.scores_csv)

    if has_box:  # split the failures: bad box (matching) vs bad mask (SAM2)
        fails = [r for r in rows if r['dice'] < args.thr and 'boxiou' in r]
        a = [r for r in fails if r['boxiou'] < args.box_thr]
        b = [r for r in fails if r['boxiou'] >= args.box_thr]
        print(f'\n=== Failure regimes (Dice < {args.thr}, boxiou < {args.box_thr}) ===')
        print(f'  A mislocation  {len(a):>3}/{len(fails)}  box on wrong tissue -> matching')
        print(f'  B growth       {len(b):>3}/{len(fails)}  box ok, mask does not grow -> SAM2')


# ============================== vis (loads MedSAM2) ==============================
# same defaults as support_prompt_for_query_dense_bodymasked_bbox
THR_HI, THR_LO = 0.7, 0.3
BODY_THRESH, BODY_MIN_PX = 10.0, 50
SCORE_THRESH, MARGIN_PX = 0.0, 0.0


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    return 1.0 if denom == 0 else float(2.0 * inter / denom)


def _dice_by_z(pred_full: np.ndarray, gt_full: np.ndarray, fg_idx: np.ndarray) -> list:
    """Per-slice Dice over the GT z-extent. A high Dice on the prompt slice next to a
    collapsing profile means the box was fine and propagation is what fails."""
    return [(int(z), _dice(pred_full[z].astype(bool), gt_full[z].astype(bool)))
            for z in fg_idx]


def _iou(pred: np.ndarray, gt: np.ndarray) -> float:
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


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


def _predict_box_support(seg, supp_slices, query_frames, n_anchors=1, min_gap=3):
    """Mirrors support_anchors_dense_bodymasked_bbox but keeps the intermediates
    (does not modify support_prompt.py). Returns (boxes, frame_idx, box, score,
    pos_map, neg_map, frame_u8): the anchor dict for segment_volume, then the
    best-scoring anchor, which is the one the figure is drawn on."""
    from models.support_prompt import (body_mask2d, build_support_bag,
                                       dense_similarity_maps, bbox_from_similarity_blob,
                                       pick_anchors)
    Pos_n, Neg_n = build_support_bag(seg, supp_slices, THR_HI, THR_LO,
                                     BODY_THRESH, BODY_MIN_PX)
    cands = []
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, BODY_THRESH, BODY_MIN_PX)
        box = bbox_from_similarity_blob(pos_map, neg_map, q_body, frame_u8.shape,
                                        SCORE_THRESH, MARGIN_PX)
        cands.append((float((pos_map - neg_map).max()), fidx, box, pos_map, neg_map, frame_u8))

    picked = pick_anchors(cands, n_anchors, min_gap)
    boxes = {c[1]: np.asarray(c[2], dtype=np.float32) for c in picked}
    best = picked[0]
    return boxes, best[1], best[2], best[0], best[3], best[4], best[5]


def _upsample(map2d: np.ndarray, hw) -> np.ndarray:
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(map2d.astype(np.float32))[None, None]
    return F.interpolate(t, size=hw, mode='bilinear', align_corners=False)[0, 0].numpy()


def _draw_box(ax, box, lw=2.5, color='cyan'):
    from matplotlib.patches import Rectangle
    x0, y0, x1, y1 = box
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           edgecolor=color, linewidth=lw))


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
    from models.support_prompt import key_slice, pick_support_slices
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
    csv_rows: list[dict] = []   # same schema as eval_medsam2.py, + boxiou for support
    z_rows: list[dict] = []     # per-slice Dice -> dice_by_z.csv (propagation profile)

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
                # full volume + absolute fg_idx, as in evaluate(); keys come out local
                boxes = _build_boxes(q_fg, fg_idx, z0, mode, args.margin)
                seg_crop = seg.segment_volume(vol_u8, boxes, refine_iters=args.refine_iters)
                pred_full = np.zeros_like(q_fg)
                pred_full[z0:z1 + 1] = seg_crop
                pb, gb = pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool)
                d, i = _dice(pb, gb), _iou(pb, gb)
                csv_rows.append({'class': label_name, 'label': label_val,
                                 'scan': qsid, 'dice': d, 'iou': i})

                frame_idx = key_slice(q_fg) - z0   # key slice is always in boxes if mode=key
                if frame_idx not in boxes:
                    frame_idx = next(iter(boxes))
                # perslice: every z is prompted, so there is nothing to propagate
                z_prompt = -1 if mode == 'perslice' else z0 + frame_idx
                for z, dz in _dice_by_z(pred_full, q_fg, fg_idx):
                    z_rows.append({'class': label_name, 'scan': qsid, 'z': z,
                                   'dice': dz, 'z_prompt': z_prompt})
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
                supp_vol_u8 = volume_to_uint8(supp_img)
                supp_zs = pick_support_slices(supp_fg, args.support_slices,
                                              args.support_min_gap)
                supp = [(supp_vol_u8[z], supp_fg[z].astype(bool)) for z in supp_zs]
                supp_z = key_slice(supp_fg)                 # the one the figure draws
                supp_frame_u8 = supp_vol_u8[supp_z]
                supp_mask2d = supp_fg[supp_z].astype(bool)

                boxes, frame_idx, box, conf, pos_map, neg_map, q_frame_u8 = _predict_box_support(
                    seg, supp, query_frames, args.n_anchors, args.anchor_min_gap)
                seg_crop = seg.segment_volume(vol_u8, boxes, refine_iters=args.refine_iters)
                pred_full = np.zeros_like(q_fg)
                pred_full[z0:z1 + 1] = seg_crop
                pb, gb = pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool)
                d, i = _dice(pb, gb), _iou(pb, gb)

                gt2d = fg_crop[frame_idx].astype(bool)
                pred2d = seg_crop[frame_idx].astype(bool)
                boxiou = _box_iou(tuple(box), _gt_bbox(gt2d))
                scan_id = f'{qsid}@{supp_sid}' if args.all_supports else qsid
                csv_rows.append({'class': label_name, 'label': label_val, 'boxiou': boxiou,
                                 'scan': scan_id, 'dice': d, 'iou': i})
                # with several anchors, "distance from the prompt" = distance to the nearest one
                anchors_z = [z0 + f for f in boxes]
                for z, dz in _dice_by_z(pred_full, q_fg, fg_idx):
                    z_rows.append({'class': label_name, 'scan': scan_id, 'z': z, 'dice': dz,
                                   'z_prompt': min(anchors_z, key=lambda a: abs(a - z))})
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
                        [f'support {supp_sid} (z={supp_z})  bag z={supp_zs}',
                         f'similarity + BOX  conf={conf:.3f}  boxiou={boxiou:.2f}',
                         f'{qsid} z={z0 + frame_idx}  anchors={len(boxes)}  Dice(vol)={d:.3f}'])
                print(f'[{label_name}] {qsid}: support={supp_sid} Dice={d:.4f} '
                      f'boxiou={boxiou:.3f} conf={conf:.3f} -> {os.path.basename(out)}')

    if not csv_rows:
        print('No scans scored — nothing written.')
        return
    fields = CSV_FIELDS + (['boxiou'] if args.box_source == 'support' else [])
    csv_path = os.path.join(args.out_dir, 'scores.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(csv_rows)

    z_path = os.path.join(args.out_dir, 'dice_by_z.csv')
    with open(z_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=Z_FIELDS)
        w.writeheader()
        w.writerows(z_rows)

    print(f'\nPer-scan scores -> {csv_path}\nPer-slice Dice  -> {z_path}\n'
          f'  triage: python3 scripts/eval/debug_medsam2.py triage {args.out_dir}')


# ============================== mcvis (B2: multiclass matching) ==============================
# Same frozen-slice protocol as `vis --query_slice key --box_source support`, but the box
# comes from cross-class competition instead of the binary pos/neg bag. Directly comparable:
# same seed, same support pairing, same query slice -> only the matching rule changes.


TYPE_COLORS = {'QF': 'tab:red', 'HS': 'tab:blue', 'SA': 'tab:green', 'GR': 'tab:orange',
              'AD': 'tab:purple', 'GLUT': 'tab:brown',
              'CALF1': 'tab:red', 'CALF2': 'tab:blue', 'CALF3': 'tab:orange',
              'CALF4': 'tab:green', 'CALF5': 'tab:purple', 'CALF6': 'tab:brown',
              'CALF7': 'tab:pink', 'CALF8': 'tab:olive', 'CALF9': 'tab:cyan'}


def _winner_map(score_maps: dict, hw: tuple) -> tuple:
    """Which type wins each pixel, and by how much. Cells no type wins (BG rival on top)
    are left as -1. This is the panel that shows who steals from whom."""
    names = sorted(score_maps)
    stack = np.stack([_upsample(score_maps[c], hw) for c in names])
    best = stack.max(0)
    win = np.where(best > 0.0, stack.argmax(0), -1)
    return win, best, names


def _draw_winner(ax, frame_u8, win, names, boxes, target: str, leg=None):
    """Winner map + every class box (target thick white, rivals thin, BG left grey).
    `leg` = the target side mask, contoured: a straight vertical edge means the legs touched
    and the midline cut fired, which is where contralateral confusion comes from."""
    from matplotlib.colors import ListedColormap, to_rgba
    ax.imshow(frame_u8, cmap='gray')
    cmap = ListedColormap([to_rgba(TYPE_COLORS.get(n, 'magenta')) for n in names])
    ax.imshow(np.ma.masked_less(win, 0), cmap=cmap, vmin=0, vmax=len(names) - 1, alpha=0.45)
    if leg is not None:
        ax.contour(leg, colors='white', linewidths=0.8, linestyles='dashed')
    for name, (_score, box, *_rest) in boxes.items():
        is_target = name == target
        _draw_box(ax, box, 2.6 if is_target else 1.0,
                  'white' if is_target else TYPE_COLORS.get(name.split('_', 1)[-1], 'magenta'))


def _draw_winner_mask(ax, frame_u8, win, names, masks, target: str, leg=None):
    """Mask-prompt sibling of _draw_winner: same categorical winner-map fill, but
    contours the pseudo-label MASK per class instead of drawing a box (target thick,
    rivals thin). Kept as a separate function (not a flag on _draw_winner) so the
    box-visualization path stays untouched."""
    from matplotlib.colors import ListedColormap, to_rgba
    ax.imshow(frame_u8, cmap='gray')
    cmap = ListedColormap([to_rgba(TYPE_COLORS.get(n, 'magenta')) for n in names])
    ax.imshow(np.ma.masked_less(win, 0), cmap=cmap, vmin=0, vmax=len(names) - 1, alpha=0.45)
    if leg is not None:
        ax.contour(leg, colors='white', linewidths=0.8, linestyles='dashed')
    for name, (_score, mask) in masks.items():
        is_target = name == target
        color = 'white' if is_target else TYPE_COLORS.get(name.split('_', 1)[-1], 'magenta')
        ax.contour(mask, colors=[color], linewidths=2.6 if is_target else 1.0)


def _render_support(out_dir: str, supp_sid: str, supp_slices: list, supp_zs: list) -> None:
    """One panel per bag slice, each type contoured in its colour: shows exactly which
    pixels feed which bag. Written once per support, not once per class."""
    def panel(frame_u8, cls_masks):
        def draw(ax, f=frame_u8, cm=cls_masks):
            ax.imshow(f, cmap='gray')
            for t, m in cm.items():
                ax.contour(m, colors=[TYPE_COLORS.get(t, 'magenta')], linewidths=1.5)
        return draw

    panels = [panel(f, cm) for f, cm in supp_slices]
    titles = [f'z={z}  ' + ' '.join(sorted(cm)) for z, (_f, cm) in zip(supp_zs, supp_slices)]
    _render(os.path.join(out_dir, f'_support_{supp_sid}.png'), panels, titles)


# _muscle_types / _muscle_types_single / _left_is_low_x / _support_bag_slices(_single) moved
# to models/support_prompt.py (shared with eval_medsam2.py's prompt_mode=support_multiclass).


def cmd_mcvis(args) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import torch
    import yaml
    from models.medsam2_adapter import MedSAM2Segmenter, volume_to_uint8
    from models.support_prompt import (body_mask2d, build_multiclass_bags, key_slice,
                                       left_is_low_x, legs_are_separate, multiclass_boxes_for_frame,
                                       multiclass_mask_anchors, multiclass_masks_for_frame,
                                       muscle_types, muscle_types_single,
                                       side_masks, support_bag_slices, support_bag_slices_single)
    from models.medsam2_adapter import mask_to_box
    from eval_medsam2 import _leg_crop_bag, _read_nii, _load_raw

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    label_names = cfg['data']['label_names']
    types = muscle_types_single(label_names) if args.single_leg else muscle_types(label_names)
    test_labels = args.test_labels or list(range(1, len(label_names)))

    paths = sorted(glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
    all_sids = [os.path.basename(p).replace('image_', '').replace('.nii.gz', '')
                for p in paths if not os.path.basename(p).startswith('image_P')]
    keep = set(all_sids if not args.only else [s for s in all_sids if s in set(args.only)])
    if not keep:
        raise ValueError('No query scans found (check --only / --target_data_dir)')

    os.makedirs(args.out_dir, exist_ok=True)
    seg = MedSAM2Segmenter(args.medsam2_ckpt, args.sam2_cfg, device=device)
    if getattr(args, 'embed_level', -1) != -1:   # A/B resolution probe: route matching to a
        _lvl = args.embed_level                   # finer FPN level (embed_frame stays untouched)
        seg.embed_frame = lambda f: seg.embed_frame_ml(f, _lvl)
        print(f'[embed_level={_lvl}] support matching uses backbone_fpn[{_lvl}] '
              f'({"128x128" if _lvl == 0 else "64x64" if _lvl == 1 else "32x32"})')
    bag_cache: dict = {}          # support sid -> (bags, left_is_low_x, slice zs)
    split_cache: dict = {}        # --split_legs: (supp_sid, side) -> (bags, left_is_low_x) | None

    csv_rows: list[dict] = []

    for label_val in test_labels:
        label_name = label_names[label_val]
        side, mtype = (None, label_name) if args.single_leg else label_name.split('_', 1)

        support_fg_idx = {}
        for sid in all_sids:
            lbl = _read_nii(os.path.join(args.target_data_dir, f'label_{sid}.nii.gz'))
            if np.where((lbl == label_val).any(axis=(1, 2)))[0].size:
                support_fg_idx[sid] = True
        rng = random.Random(args.seed + label_val)

        for qsid in all_sids:   # iterate all: keeps the rng sequence == evaluate()'s
            q_img, q_lbl = _load_raw(args.target_data_dir, qsid)
            q_fg = (q_lbl == label_val).astype(np.uint8)
            fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]

            pool = [s for s in support_fg_idx if s != qsid]
            if len(fg_idx) == 0 or not pool:
                continue
            supp_sid = rng.choice(pool)
            if qsid not in keep:
                continue

            if supp_sid not in bag_cache:
                supp_img, supp_lbl = _load_raw(args.target_data_dir, supp_sid)
                supp_vol_u8 = volume_to_uint8(supp_img)
                bag_slices_fn = support_bag_slices_single if args.single_leg else support_bag_slices
                supp_slices, supp_zs = bag_slices_fn(
                    supp_vol_u8, supp_lbl, types, args.support_slices, args.support_min_gap)
                low_x = None if args.single_leg else left_is_low_x(supp_lbl, types)
                bag_cache[supp_sid] = (build_multiclass_bags(seg, supp_slices, THR_HI, THR_LO,
                                                             BODY_THRESH, BODY_MIN_PX),
                                       low_x, supp_zs)
                _render_support(args.out_dir, supp_sid, supp_slices, supp_zs)
            bags, low_x, supp_zs = bag_cache[supp_sid]

            z0, z1 = int(fg_idx.min()), int(fg_idx.max())
            vol_u8 = volume_to_uint8(q_img)[z0:z1 + 1]

            # --split_legs: crop query+support to one leg, reuse the single_leg=True
            # pipeline unmodified on the crop (see eval_medsam2.py:_leg_crop_bag /
            # models/support_prompt.py:leg_crop_boxes). Viz panels below are drawn on
            # the CROPPED frame in that case, not the full bilateral canvas.
            split_active = args.split_legs and not args.single_leg and side is not None
            crop_box = None
            if split_active:
                leg = _leg_crop_bag(seg, args.target_data_dir, supp_sid, vol_u8, types,
                                    label_name, args.support_slices, args.support_min_gap,
                                    split_cache)
                if leg is None:
                    print(f'[{label_name}] {qsid}: could not isolate the {side} leg '
                          f'(split_legs)')
                    continue
                bags_v, vol_u8_v, crop_box, _mtype_v = leg
                low_x_v, single_leg_v = None, True
            else:
                bags_v, vol_u8_v, low_x_v, single_leg_v = bags, vol_u8, low_x, args.single_leg

            frame_idx = key_slice(q_fg) - z0             # frozen slice, as in bag_key
            frame_u8 = vol_u8_v[frame_idx]
            gt2d_full = q_fg[z0 + frame_idx].astype(bool)
            if split_active:
                y0, y1, x0, x1 = crop_box
                gt2d = gt2d_full[y0:y1, x0:x1]
            else:
                gt2d = gt2d_full

            # prompts/score_maps are keyed by bare type name (mtype) in single_leg/
            # split_legs mode, by '<side>_<type>' (label_name) otherwise -- pkey picks
            # the right one for dict lookups; mtype (score_maps, legend) is unaffected,
            # multiclass_score_maps only ever sees bare types (support_bag_slices pools
            # L+R into one bag per type regardless of split_legs).
            pkey = mtype if (args.single_leg or split_active) else label_name

            mask_mode = args.prompt_mode == 'support_multiclass_mask'
            if mask_mode:
                prompts, score_maps = multiclass_masks_for_frame(
                    seg, bags_v, frame_u8, low_x_v, BODY_THRESH, BODY_MIN_PX,
                    SCORE_THRESH, single_leg=single_leg_v, cc_mode=args.cc_mode)
            else:
                prompts, score_maps = multiclass_boxes_for_frame(
                    seg, bags_v, frame_u8, low_x_v, BODY_THRESH, BODY_MIN_PX,
                    SCORE_THRESH, MARGIN_PX, single_leg=single_leg_v, cc_mode=args.cc_mode,
                    neg_points=args.neg_points, max_neg_points=args.max_neg_points)

            # mask_mode + n_anchors>1: search every query FG slice instead of trusting the
            # single frozen key slice above; rescues the class when it lost to a rival there
            # but survives elsewhere. n_anchors=1 (default) keeps the key-slice-only prompt,
            # so viz panels (which are drawn on frame_u8 = key slice) stay unchanged.
            mask_anchors: dict = {}
            if mask_mode:
                if args.n_anchors > 1:
                    cand_frames = [(int(z) - z0, vol_u8_v[int(z) - z0]) for z in fg_idx]
                    mask_anchors = multiclass_mask_anchors(
                        seg, bags_v, cand_frames, pkey, low_x_v,
                        n_anchors=args.n_anchors, min_gap=args.anchor_min_gap,
                        single_leg=single_leg_v, cc_mode=args.cc_mode)
                elif pkey in prompts:
                    mask_anchors = {frame_idx: prompts[pkey][1]}

            win, best, names = _winner_map(score_maps, frame_u8.shape)
            score_up = _upsample(score_maps[mtype], frame_u8.shape)
            owned = float((win == names.index(mtype)).mean())   # frame share this type claims

            body2d = body_mask2d(frame_u8, BODY_THRESH, BODY_MIN_PX)
            if split_active:
                leg2d, split = body2d, f'split:{side}'   # already cropped to one leg
            elif args.single_leg:
                leg2d, split = body2d, 'n/a'    # no side to split -- whole body is the group
            else:
                leg2d = side_masks(body2d, low_x)[side]
                split = 'cc' if legs_are_separate(body2d) else 'MIDLINE'

            def w(ax, f=frame_u8, wn=win, nm=names, pr=prompts, t=pkey, g=gt2d, lg=leg2d):
                (_draw_winner_mask if mask_mode else _draw_winner)(ax, f, wn, nm, pr, t, lg)
                ax.contour(g, colors='yellow', linewidths=1.5)

            def sc(ax, f=frame_u8, s=score_up, g=gt2d):
                ax.imshow(f, cmap='gray'); ax.imshow(s, cmap='jet', alpha=0.45)
                ax.contour(g, colors='yellow', linewidths=1.5)

            legend = ' '.join(f'{n}={TYPE_COLORS.get(n, "?").replace("tab:", "")}'
                              for n in names)
            t_win = f'winner  supp={supp_sid} z={supp_zs}  split={split}\n{legend}'
            t_sc = f'score[{mtype}] - max(rivals)   claims {owned:.1%} of frame'

            lost = (not mask_anchors) if mask_mode else (pkey not in prompts)
            if lost:      # every cell of this leg lost to a rival on every candidate slice
                csv_rows.append({'class': label_name, 'label': label_val, 'scan': qsid,
                                 'dice': 0.0, 'iou': 0.0, 'boxiou': 0.0, 'split': split})
                tag = 'NOMASK' if mask_mode else 'NOBOX'
                out = os.path.join(args.out_dir, f'{label_name}_{qsid}_{tag}.png')
                _render(out, [w, sc], [t_win, t_sc])
                print(f'[{label_name}] {qsid}: support={supp_sid} {tag} (lost to a rival) '
                      f'split={split} -> {os.path.basename(out)}')
                continue

            if mask_mode:
                # prompt_mask (cyan contour) is drawn on frame_u8 = key slice: only
                # available when the key slice itself is one of the chosen anchors.
                # conf falls back to nan when n_anchors rescued the class on a slice
                # other than the key one (multiclass_mask_anchors doesn't carry scores out).
                conf = prompts[pkey][0] if pkey in prompts else float('nan')
                prompt_mask = mask_anchors.get(frame_idx)
                npts = []
                seg_crop_v = seg.segment_volume_mask(vol_u8_v, mask_anchors)
                box_for_iou = mask_to_box(prompt_mask) if prompt_mask is not None else None
            elif args.neg_points:
                conf, box, npts = prompts[pkey]
                seg_crop_v = seg.segment_volume(
                    vol_u8_v, {frame_idx: np.asarray(box, np.float32)},
                    refine_iters=args.refine_iters,
                    neg_points={frame_idx: npts} if npts else None)
                box_for_iou = tuple(box)
            else:
                conf, box = prompts[pkey]
                npts = []
                seg_crop_v = seg.segment_volume(
                    vol_u8_v, {frame_idx: np.asarray(box, np.float32)},
                    refine_iters=args.refine_iters, neg_points=None)
                box_for_iou = tuple(box)

            pred_full = np.zeros_like(q_fg)
            if split_active:
                seg_crop = np.zeros_like(vol_u8, dtype=seg_crop_v.dtype)
                seg_crop[:, y0:y1, x0:x1] = seg_crop_v
            else:
                seg_crop = seg_crop_v
            pred_full[z0:z1 + 1] = seg_crop
            pb, gb = pred_full[fg_idx].astype(bool), q_fg[fg_idx].astype(bool)
            d, i = _dice(pb, gb), _iou(pb, gb)
            boxiou = _box_iou(box_for_iou, _gt_bbox(gt2d)) if box_for_iou is not None else 0.0
            csv_rows.append({'class': label_name, 'label': label_val, 'scan': qsid,
                             'dice': d, 'iou': i, 'boxiou': boxiou, 'split': split})
            pred2d = seg_crop_v[frame_idx].astype(bool)   # cropped coords, matches frame_u8

            if mask_mode:
                def m1(ax, f=frame_u8, g=gt2d, pr=pred2d, pm=prompt_mask):
                    ax.imshow(f, cmap='gray'); ax.contour(g, colors='yellow', linewidths=1.5)
                    ax.contour(pr, colors='red', linewidths=1.5)
                    if pm is not None:   # key slice not among the chosen anchors (rescued
                        ax.contour(pm, colors='cyan', linewidths=2.0)   # elsewhere by n_anchors)
            else:
                def m1(ax, f=frame_u8, g=gt2d, pr=pred2d, b=box, p=npts):
                    ax.imshow(f, cmap='gray'); ax.contour(g, colors='yellow', linewidths=1.5)
                    ax.contour(pr, colors='red', linewidths=1.5); _draw_box(ax, b, 2.0)
                    if p:
                        xs, ys = zip(*p)
                        ax.scatter(xs, ys, c='magenta', marker='x', s=60, linewidths=2)

            out = os.path.join(args.out_dir,
                               f'{label_name}_{qsid}_dice{d:.3f}_boxiou{boxiou:.2f}.png')
            _render(out, [w, sc, m1], [t_win, t_sc,
                    f'{qsid} [{label_name}] side={side}  Dice(vol)={d:.3f} boxiou={boxiou:.2f}'])
            anchors_note = (f' anchors_z={sorted(z0 + f for f in mask_anchors)}'
                            if mask_mode and args.n_anchors > 1 else '')
            print(f'[{label_name}] {qsid}: support={supp_sid} Dice={d:.4f} '
                  f'boxiou={boxiou:.3f} conf={conf:.3f} claims={owned:.1%} '
                  f'split={split}{anchors_note} -> {os.path.basename(out)}')

    if not csv_rows:
        print('No scans scored — nothing written.')
        return
    csv_path = os.path.join(args.out_dir, 'scores.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS + ['boxiou', 'split'])
        w.writeheader()
        w.writerows(csv_rows)

    if args.single_leg:
        split_line = '  single_leg: no L/R split (whole body mask used as one group)\n'
    else:
        n_mid = sum(r['split'] == 'MIDLINE' for r in csv_rows)
        split_line = f'  leg split: {len(csv_rows) - n_mid} cc, {n_mid} midline (touching legs)\n'
    print(f'\nPer-scan scores -> {csv_path}\n{split_line}'
          f'  triage: python3 scripts/eval/debug_medsam2.py triage {args.out_dir}')


# ============================== CLI ==============================


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('triage', help='Dice/IoU per class from any run: a scores.csv or its dir')
    t.add_argument('scores_csv', help='path to scores.csv, or the run dir holding it')
    t.add_argument('--thr', type=float, default=0.40, help='Dice below this = failed scan')
    t.add_argument('--box_thr', type=float, default=0.10,
                   help='support runs: boxiou below this = mislocation (Regime A)')
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
    v.add_argument('--n_anchors', type=int, default=1,
                   help='box_source=support: prompt the box on the N best-scoring slices')
    v.add_argument('--anchor_min_gap', type=int, default=3,
                   help='min z-distance between anchors (--n_anchors > 1)')
    v.add_argument('--support_slices', type=int, default=1,
                   help='box_source=support: B1, build the Pos/Neg bag from K support slices')
    v.add_argument('--support_min_gap', type=int, default=3,
                   help='min z-distance between support slices (--support_slices > 1)')
    v.add_argument('--refine_iters', type=int, default=1)
    v.add_argument('--seed', type=int, default=42, help='must match the eval to reproduce pairings')
    v.add_argument('--only', nargs='+', default=None, help='limit to these query sids')
    v.add_argument('--all_supports', action='store_true',
                   help='box_source=support only: draw EVERY candidate support (variance)')
    v.add_argument('--device', default=None)
    v.add_argument('--out_dir', required=True)
    v.set_defaults(func=cmd_vis)

    m = sub.add_parser('mcvis', help='B2: box from cross-class competition (frozen slice)')
    m.add_argument('--config', required=True)
    m.add_argument('--medsam2_ckpt', required=True)
    m.add_argument('--sam2_cfg', required=True)
    m.add_argument('--target_data_dir', required=True)
    m.add_argument('--test_labels', type=int, nargs='+', default=None)
    m.add_argument('--support_slices', type=int, default=3,
                   help='K slices per muscle type in the support bags (B1)')
    m.add_argument('--support_min_gap', type=int, default=3)
    m.add_argument('--refine_iters', type=int, default=1)
    m.add_argument('--seed', type=int, default=42, help='must match the eval to reproduce pairings')
    m.add_argument('--only', nargs='+', default=None)
    m.add_argument('--device', default=None)
    m.add_argument('--out_dir', required=True)
    m.add_argument('--single_leg', action='store_true',
                   help='dataset has one leg per volume (no L/R): label_names are bare '
                        'type names, no side split of the body mask')
    m.add_argument('--split_legs', action='store_true',
                   help='(bilateral datasets only, ignored with --single_leg) crop query + '
                        'support to one leg and run the single-leg pipeline on each side '
                        'independently instead of competing both legs in the same fixed '
                        '512x512 SAM2 grid -- fixes a resolution deficit that starves small/'
                        'thin muscles (SA, GR). Viz panels are drawn on the cropped frame. '
                        'See models/support_prompt.py:leg_crop_boxes')
    m.add_argument('--cc_mode', choices=['dilate_largest', 'union', 'seed_only'],
                   default='dilate_largest',
                   help='_box_from_blob CC selection: dilate_largest = current fix, '
                        'union = superseded first fix, seed_only = pre-fix ablation baseline')
    m.add_argument('--neg_points', action='store_true',
                   help='box+neg-points hybrid: add negative clicks on rival-won cells '
                        'inside the box (magenta x in the rendered PNG), to push SAM2 off '
                        'neighboring muscle an elongated box also covers (e.g. soleus)')
    m.add_argument('--max_neg_points', type=int, default=3,
                   help='(--neg_points) cap on negative clicks per box')
    m.add_argument('--prompt_mode', choices=['support_multiclass', 'support_multiclass_mask'],
                   default='support_multiclass',
                   help='support_multiclass = box from the winner-take-all blob (default, '
                        'previous behavior); support_multiclass_mask = raw blob fed to SAM2 '
                        'as a mask prompt instead (models/support_prompt.py multiclass_masks_'
                        'for_frame, models/medsam2_adapter.py segment_volume_mask). refine_iters '
                        'and neg_points are box-only and ignored in mask mode.')
    m.add_argument('--n_anchors', type=int, default=1,
                   help='(prompt_mode=support_multiclass_mask only; box mode stays frozen-'
                        'slice) search every query FG slice instead of the single key slice, '
                        'prompt the N where the class survives the rival winner-take-all -- '
                        'rescues the catastrophic-zero case (class lost on the key slice '
                        'alone). 1 = previous single-key-slice behavior, same viz panels.')
    m.add_argument('--anchor_min_gap', type=int, default=3,
                   help='min z-distance between anchors (--n_anchors > 1)')
    m.add_argument('--embed_level', type=int, default=-1, choices=[-1, 0, 1, 2],
                   help='A/B resolution probe: backbone_fpn level for support matching. '
                        '-1=stride16/32x32 (default, production embed_frame), 1=stride8/64x64, '
                        '0=stride4/128x128 (finest, lateral-only). Run each level to its own '
                        '--out_dir and compare boxiou: finer fixes mislocation = Regime A '
                        '(resolution); still broken = Regime B (textural confusion).')
    m.set_defaults(func=cmd_mcvis)

    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
