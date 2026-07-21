"""
PerSAM-style one-shot BOX prompting from a support volume+mask, using SAM2's own image
encoder. Used by scripts/eval/eval_medsam2.py (prompt_mode=support_bbox).

Point-prompt variants were removed once box prompting superseded them (R_HS, 20 HV scans:
point Dice 0.3673 vs box 0.6370). See git history.
"""
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F


def key_slice(fg_mask: np.ndarray) -> int:
    """fg_mask: [Z,H,W]. Returns z-index of the max-FG-area slice."""
    areas = fg_mask.reshape(fg_mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def pick_support_slices(fg_mask: np.ndarray, n_slices: int = 1,
                        min_gap: int = 3) -> list:
    """fg_mask: [Z,H,W]. B1: the K slices whose vectors go into the support bag. Same
    greedy as pick_anchors, scored by FG area instead of matching score, so the slices
    are spread along z and cover the shape variation, not just the fattest section.
    n_slices=1 == key_slice. returns: sorted z-indices."""
    areas = fg_mask.reshape(fg_mask.shape[0], -1).sum(axis=1)
    cands = [(float(a), int(z)) for z, a in enumerate(areas) if a > 0]
    return sorted(z for _, z in pick_anchors(cands, n_slices, min_gap))


def body_mask2d(img_u8: np.ndarray, thresh: float = 10.0,
                 min_component_px: int = 50) -> np.ndarray:
    """img_u8: [H,W] uint8. Thresholds out air/padding, keeps EVERY connected component
    >= min_component_px (bilateral anatomy: two legs = two blobs; keeping only the
    largest silently drops one). Fills holes. Returns [H,W] bool."""
    from scipy.ndimage import binary_fill_holes
    from skimage.measure import label as cc_label

    m = img_u8 > thresh
    if not m.any():
        return m
    labeled = cc_label(m)
    sizes = np.bincount(labeled.flat)
    keep = np.zeros_like(m)
    for comp_id, size in enumerate(sizes):
        if comp_id == 0:
            continue
        if size >= min_component_px:
            keep |= (labeled == comp_id)
    return binary_fill_holes(keep)


def extract_support_vectors_bodymasked(feat: torch.Tensor, mask2d: np.ndarray,
                                        supp_body2d: np.ndarray, thr_hi: float = 0.7,
                                        thr_lo: float = 0.3) -> tuple:
    """Bag-of-vectors (not a single averaged prototype) from the support's encoder feature.
    FG bag = cells with mask > thr_hi; BG bag = cells with mask < thr_lo AND inside the
    body mask. The ambiguous boundary band in between is dropped.
    Returns (Pos_n [C,Np], Neg_n [C,Nn]), L2-normalized; either may be empty."""
    C, h, w = feat.shape
    m = torch.from_numpy(mask2d.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    m = m.to(feat.device)

    b = torch.from_numpy(supp_body2d.astype(np.float32))[None, None]
    b = F.interpolate(b, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    b = b.to(feat.device)

    feat_flat = feat.reshape(C, h * w)
    mask_flat = m.reshape(h * w)
    body_flat = b.reshape(h * w)

    pos_idx = (mask_flat > thr_hi).nonzero(as_tuple=True)[0]
    neg_idx = ((mask_flat < thr_lo) & (body_flat > 0.5)).nonzero(as_tuple=True)[0]

    Pos = feat_flat[:, pos_idx]
    Neg = feat_flat[:, neg_idx]
    Pos_n = F.normalize(Pos, dim=0) if Pos.shape[1] > 0 else Pos
    Neg_n = F.normalize(Neg, dim=0) if Neg.shape[1] > 0 else Neg
    return Pos_n, Neg_n


def dense_similarity_maps(feat_query: torch.Tensor, Pos_n: torch.Tensor,
                           Neg_n: torch.Tensor) -> tuple:
    """Per query cell, max cosine similarity against the whole positive bag and, separately,
    the whole negative bag (nearest-neighbor matching). Returns (pos_map, neg_map) [h,w]; an
    empty bag yields an all -1 map."""
    C, h, w = feat_query.shape
    Q_n = F.normalize(feat_query.reshape(C, h * w), dim=0)  # [C,M]

    def _max_sim(Bag_n):
        if Bag_n.shape[1] == 0:
            return -np.ones((h, w), dtype=np.float32)
        sim = Bag_n.t() @ Q_n                 # [N,M]
        return sim.max(dim=0).values.reshape(h, w).cpu().numpy()

    return _max_sim(Pos_n), _max_sim(Neg_n)


def bbox_from_similarity_blob(pos_map: np.ndarray, neg_map: np.ndarray,
                               query_body2d: np.ndarray, img_hw: tuple,
                               score_thresh: float = 0.0, margin_px: float = 0.0) -> tuple:
    """Box prompt from the score_map (= pos_map - neg_map) object blob: connected component
    of {score_map > score_thresh} containing its argmax, clipped to the query body mask.
    Returns (x0,y0,x1,y1) in original (H,W) pixels. Gives SAM2 the object's extent directly
    instead of relying on it to grow a mask from a point. No query GT is read."""
    from skimage.measure import label as cc_label

    h, w = pos_map.shape
    H, W = img_hw

    def cell_to_xy(row, col):
        return (float(col) / w * W, float(row) / h * H)

    score_map = pos_map - neg_map
    pos_idx = np.unravel_index(np.argmax(score_map), score_map.shape)

    blob_seed = score_map > score_thresh
    labeled = cc_label(blob_seed)
    comp_id = labeled[pos_idx]
    if comp_id != 0:
        blob = (labeled == comp_id)
    else:
        blob = np.zeros_like(blob_seed)
        blob[pos_idx] = True

    body = torch.from_numpy(query_body2d.astype(np.float32))[None, None]
    body = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    body_grid = (body.numpy() > 0.5)

    blob_in_body = blob & body_grid
    if not blob_in_body.any():   # blob entirely outside the body mask
        blob_in_body = blob

    ys, xs = np.where(blob_in_body)
    y0, x0 = ys.min(), xs.min()
    y1, x1 = ys.max() + 1, xs.max() + 1
    px0, py0 = cell_to_xy(y0, x0)
    px1, py1 = cell_to_xy(y1, x1)
    px0 = max(0.0, px0 - margin_px)
    py0 = max(0.0, py0 - margin_px)
    px1 = min(float(W), px1 + margin_px)
    py1 = min(float(H), py1 + margin_px)
    return (px0, py0, px1, py1)


def build_support_bag(seg, supp_slices: list, thr_hi: float = 0.7, thr_lo: float = 0.3,
                       body_thresh: float = 10.0, body_min_px: int = 50) -> tuple:
    """B1: one Pos/Neg bag from K slices of the SAME support volume (still 1-shot).
    supp_slices: [(frame_u8, mask2d)]. Columns are already L2-normalized per slice, so
    concatenating needs no renormalization. K=1 reproduces the single-key-slice bag.
    Returns (Pos_n [C,Np], Neg_n [C,Nn])."""
    pos, neg, C, device = [], [], None, None
    for frame_u8, mask2d in supp_slices:
        feat = seg.embed_frame(frame_u8)
        C, device = feat.shape[0], feat.device
        body = body_mask2d(frame_u8, body_thresh, body_min_px)
        P, N = extract_support_vectors_bodymasked(feat, mask2d, body, thr_hi, thr_lo)
        if P.shape[1] > 0:
            pos.append(P)
        if N.shape[1] > 0:
            neg.append(N)

    def _cat(bags):
        return torch.cat(bags, dim=1) if bags else torch.zeros((C, 0), device=device)

    return _cat(pos), _cat(neg)


def score_query_frames(seg, supp_slices: list, query_frames: list,
                        thr_hi: float = 0.7, thr_lo: float = 0.3,
                        body_thresh: float = 10.0, body_min_px: int = 50,
                        score_thresh: float = 0.0, margin_px: float = 0.0) -> list:
    """seg: MedSAM2Segmenter (only seg.embed_frame is used).
    supp_slices: [(frame_u8, mask2d)] support slices + their GT masks.
    query_frames: list[(frame_idx, frame_u8)] candidates.

    One box per candidate frame, from that frame's similarity blob. No query GT read.
    returns: [(score, frame_idx, box_xyxy)] in query_frames order.
    """
    Pos_n, Neg_n = build_support_bag(seg, supp_slices, thr_hi, thr_lo,
                                     body_thresh, body_min_px)

    cands = []
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, body_thresh, body_min_px)
        box = bbox_from_similarity_blob(pos_map, neg_map, q_body, frame_u8.shape,
                                        score_thresh, margin_px)
        cands.append((float((pos_map - neg_map).max()), fidx, box))
    return cands


def pick_anchors(cands: list, n_anchors: int = 1, min_gap: int = 3) -> list:
    """cands: [(score, frame_idx, ...)], extra fields ignored. Greedy: take the highest
    score, then the next one at least min_gap slices away, up to n_anchors. Spreads the
    prompts along z instead of clustering them on one confident slice.
    n_anchors=1 == plain argmax (sort is stable, so ties break as in the argmax loop).
    returns: the selected candidates, score-descending."""
    picked: list = []
    for c in sorted(cands, key=lambda c: -c[0]):
        if len(picked) >= n_anchors:
            break
        if all(abs(c[1] - p[1]) >= min_gap for p in picked):
            picked.append(c)
    return picked


def support_prompt_for_query_dense_bodymasked_bbox(seg, supp_slices: list,
                                                    query_frames: list,
                                                    thr_hi: float = 0.7, thr_lo: float = 0.3,
                                                    body_thresh: float = 10.0,
                                                    body_min_px: int = 50,
                                                    score_thresh: float = 0.0,
                                                    margin_px: float = 0.0) -> tuple:
    """Single-prompt case: the query frame with the highest max(pos_map - neg_map).

    returns: (frame_idx, box_xyxy)
    """
    cands = score_query_frames(seg, supp_slices, query_frames, thr_hi,
                                thr_lo, body_thresh, body_min_px, score_thresh, margin_px)
    _, fidx, box = pick_anchors(cands, n_anchors=1)[0]
    return fidx, box


def support_anchors_dense_bodymasked_bbox(seg, supp_slices: list, query_frames: list,
                                           n_anchors: int = 1, min_gap: int = 3,
                                           thr_hi: float = 0.7, thr_lo: float = 0.3,
                                           body_thresh: float = 10.0,
                                           body_min_px: int = 50,
                                           score_thresh: float = 0.0,
                                           margin_px: float = 0.0) -> dict:
    """Multi-prompt (B4): re-anchor the box on up to n_anchors slices instead of one.
    SAM2 then propagates from every anchor, so a bad box no longer sinks the whole volume
    and memory attention is refreshed before the object is lost. Boxes for all candidate
    frames are computed anyway, so this costs no extra encoder passes.

    returns: {frame_idx -> box_xyxy float32}, ready for MedSAM2Segmenter.segment_volume.
    """
    cands = score_query_frames(seg, supp_slices, query_frames, thr_hi,
                                thr_lo, body_thresh, body_min_px, score_thresh, margin_px)
    return {fidx: np.asarray(box, dtype=np.float32)
            for _, fidx, box in pick_anchors(cands, n_anchors, min_gap)}


def support_prompt_for_query_dense_bodymasked_bbox_consensus(seg, supp_slices: list,
                                                              query_frames: list,
                                                              thr_hi: float = 0.7, thr_lo: float = 0.3,
                                                              body_thresh: float = 10.0,
                                                              body_min_px: int = 50,
                                                              score_thresh: float = 0.0,
                                                              margin_px: float = 0.0,
                                                              consensus_k: int = 5) -> tuple:
    """Same as ..._bbox, but instead of the winner-take-all frame (global argmax score,
    occasionally won by a confident false match on bone marrow), picks among the top
    consensus_k frames the one whose box center is closest to their median center -- a
    spurious match is usually a spatial outlier. Mixed A/B results, NOT wired into
    eval_medsam2.py; kept as a separate call path.

    returns: (frame_idx, box_xyxy)
    """
    candidates = [(score, fidx, box, ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0))
                  for score, fidx, box in
                  score_query_frames(seg, supp_slices, query_frames, thr_hi,
                                     thr_lo, body_thresh, body_min_px, score_thresh, margin_px)]
    candidates.sort(key=lambda c: -c[0])
    top = candidates[:max(1, min(consensus_k, len(candidates)))]
    med_cx = float(np.median([c[3][0] for c in top]))
    med_cy = float(np.median([c[3][1] for c in top]))

    def dist_to_median(c):
        cx, cy = c[3]
        return (cx - med_cx) ** 2 + (cy - med_cy) ** 2

    winner = min(top, key=dist_to_median)
    return winner[1], winner[2]


# ================================ B2: multiclass matching ================================
# The binary Neg bag is one max over the whole body (thousands of columns), so neg_map is
# nearly constant and score = pos - neg degenerates to pos: nothing ever says "that is GR,
# not SA". Here every muscle TYPE gets its own bag and they compete cell by cell.
# Types, not the 8 labels: L_QF and R_QF are indistinguishable in feature space, so making
# them rivals cancels both scores. Side is recovered spatially (the two legs are already
# two connected components of the body mask).

BG_KEY = 'BG'


def _to_grid(mask2d: np.ndarray, h: int, w: int, device=None) -> torch.Tensor:
    t = torch.from_numpy(mask2d.astype(np.float32))[None, None]
    g = F.interpolate(t, size=(h, w), mode='bilinear', align_corners=False)[0, 0]
    return g.to(device) if device is not None else g


def build_multiclass_bags(seg, supp_slices: list, thr_hi: float = 0.7, thr_lo: float = 0.3,
                          body_thresh: float = 10.0, body_min_px: int = 50) -> dict:
    """supp_slices: [(frame_u8, {cls: mask2d})], cls = muscle type (L+R pooled).
    One L2-normalized bag per type + a BG bag (body minus every type: bone, fat, skin).
    Returns {cls: [C,N]}, BG_KEY included."""
    bags = defaultdict(list)
    for frame_u8, cls_masks in supp_slices:
        feat = seg.embed_frame(frame_u8)
        C, h, w = feat.shape
        flat = feat.reshape(C, h * w)
        body = _to_grid(body_mask2d(frame_u8, body_thresh, body_min_px), h, w, feat.device)
        is_bg = body.reshape(-1) > 0.5

        for cls, m2d in cls_masks.items():
            m = _to_grid(m2d, h, w, feat.device).reshape(-1)
            idx = (m > thr_hi).nonzero(as_tuple=True)[0]
            if idx.numel():
                bags[cls].append(F.normalize(flat[:, idx], dim=0))
            is_bg &= (m < thr_lo)

        idx = is_bg.nonzero(as_tuple=True)[0]
        if idx.numel():
            bags[BG_KEY].append(F.normalize(flat[:, idx], dim=0))

    return {c: torch.cat(v, dim=1) for c, v in bags.items()}


def multiclass_score_maps(feat_query: torch.Tensor, bags: dict) -> dict:
    """score_c(x) = pos_c(x) - max over the rival bags (other types AND BG). The rival is
    an explicit balanced bag, not a saturated max over the whole body, so the contrast
    actually discriminates. Returns {cls: [h,w]} for the muscle types only."""
    C, h, w = feat_query.shape
    Qn = F.normalize(feat_query.reshape(C, h * w), dim=0)

    def _max_sim(B):
        sim = B.t() @ Qn                      # [N,M]
        return sim.max(dim=0).values.reshape(h, w)

    pos = {c: _max_sim(B) for c, B in bags.items() if B.shape[1] > 0}

    out = {}
    for c in pos:
        if c == BG_KEY:
            continue
        rivals = torch.stack([p for k, p in pos.items() if k != c])
        out[c] = (pos[c] - rivals.max(dim=0).values).cpu().numpy()
    return out


def _split_at_midline(body2d: np.ndarray) -> tuple:
    """Cut the body in two at the column with the fewest body pixels, searched around the
    centroid: on an axial thigh that column is the gap between the two legs. Used when the
    legs touch, where a connected-component split would return one blob."""
    cols = body2d.sum(axis=0).astype(np.float64)
    w = body2d.shape[1]
    cx = int(np.average(np.arange(w), weights=cols)) if cols.sum() else w // 2

    lo, hi = max(1, cx - w // 6), min(w - 1, cx + w // 6 + 1)
    cut = lo + int(np.argmin(cols[lo:hi])) if hi > lo else cx

    left, right = body2d.copy(), body2d.copy()
    left[:, cut:] = False
    right[:, :cut] = False
    return left, right


def _two_legs_cc(body2d: np.ndarray, min_leg_ratio: float = 0.2):
    """The two legs as connected components, ordered by x. None when they touch (one
    component) or when the second component is too small to be a leg (a coil, a marker)."""
    from skimage.measure import label as cc_label

    lab = cc_label(body2d)
    sizes = np.bincount(lab.flat)
    sizes[0] = 0
    comps = [c for c in np.argsort(sizes)[::-1][:2] if sizes[c] > 0]
    if len(comps) < 2 or sizes[comps[1]] < min_leg_ratio * sizes[comps[0]]:
        return None

    a, b = sorted(comps, key=lambda c: np.where(lab == c)[1].mean())
    return lab == a, lab == b


def legs_are_separate(body2d: np.ndarray, min_leg_ratio: float = 0.2) -> bool:
    """True when the CC split is trustworthy. False = side_masks used the midline cut."""
    return _two_legs_cc(body2d, min_leg_ratio) is not None


def side_masks(body2d: np.ndarray, left_is_low_x: bool,
               min_leg_ratio: float = 0.2) -> dict:
    """Split the body mask into the two legs. Two comparable connected components = the two
    legs; otherwise fall back to a midline cut -- never to the whole body for both sides,
    which would hand L and R the same box. Returns {'L': mask, 'R': mask}."""
    legs = _two_legs_cc(body2d, min_leg_ratio)
    lo_m, hi_m = legs if legs is not None else _split_at_midline(body2d)
    l, r = (lo_m, hi_m) if left_is_low_x else (hi_m, lo_m)
    return {'L': l, 'R': r}


def leg_crop_boxes(vol_u8: np.ndarray, left_is_low_x: bool, margin_frac: float = 0.15,
                   min_leg_ratio: float = 0.2) -> dict:
    """{'L': (y0,y1,x0,x1), 'R': (y0,y1,x0,x1)} pixel windows, one per leg, valid across
    the WHOLE z-range of vol_u8 (union of each frame's side_masks bbox + margin) -- SAM2
    propagation needs a fixed crop window across frames, can't shift per-slice. Side
    missing from every frame (e.g. a single-leg volume slipped in) is simply absent from
    the returned dict.

    Restores per-leg resolution before SAM2's fixed 512x512 resize (medsam2_adapter.py
    IMG_SIZE): a bilateral scan crams both legs into the same coarse embedding grid a
    single-leg scan gives entirely to one leg -- on MRI_muscle (bilateral, 560x560) vs
    MRI_muscle_2 (single-leg, 282x339) that works out to roughly 1.6x less effective
    per-leg resolution, which starves already-small/thin muscles (SA, GR) disproportio-
    nately (mean Dice ~0.72 single-leg vs 0.28-0.50 bilateral for the same types, same
    4-way winner-take-all -- support_bag_slices already pools L+R into one bag per type,
    so the competition itself was never 8-way). Used by eval_medsam2.py/debug_medsam2.py
    --split_legs: crop query+support to one leg, run the existing single_leg=True
    pipeline (unmodified) on each side independently, paste the result back.
    """
    H, W = vol_u8.shape[1], vol_u8.shape[2]
    acc: dict = {'L': None, 'R': None}
    for z in range(vol_u8.shape[0]):
        body = body_mask2d(vol_u8[z])
        if not body.any():
            continue
        for side, smask in side_masks(body, left_is_low_x, min_leg_ratio).items():
            if not smask.any():
                continue
            ys, xs = np.where(smask)
            y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
            if acc[side] is None:
                acc[side] = [y0, y1, x0, x1]
            else:
                acc[side][0] = min(acc[side][0], y0)
                acc[side][1] = max(acc[side][1], y1)
                acc[side][2] = min(acc[side][2], x0)
                acc[side][3] = max(acc[side][3], x1)

    out = {}
    for side, b in acc.items():
        if b is None:
            continue
        y0, y1, x0, x1 = b
        mh = int(round((y1 - y0) * margin_frac))
        mw = int(round((x1 - x0) * margin_frac))
        out[side] = (max(0, y0 - mh), min(H, y1 + mh + 1),
                     max(0, x0 - mw), min(W, x1 + mw + 1))
    return out


def _box_from_blob(blob: np.ndarray, score: np.ndarray, img_hw: tuple,
                   margin_px: float = 0.0, dilate_iters: int = 1,
                   cc_mode: str = 'dilate_largest', min_cc_px: int = 2) -> tuple:
    """blob -> box in (H,W) px. cc_mode picks which cells of blob feed the box:

    - 'dilate_largest' (default, current fix): dilate blob by dilate_iters cells first (to
      merge an anatomical region's own separate coarse-grid pieces -- e.g. HS's 3 heads,
      sitting a cell or two apart at this resolution -- without a distant noise speck a few
      cells away also merging in), take the largest CC of the dilated mask, crop back to
      blob pixels. Dilation never inflates the box directly since selection is intersected
      with the undilated blob.
    - 'union' (superseded first fix): union of every CC with area >= min_cc_px, no dilation.
      Over-corrects: a lone stray winning cell far from the real region (e.g. QF_06: a
      couple of QF-won cells near the AD region) balloons the box across the whole leg.
    - 'seed_only' (pre-fix / ablation baseline): CC containing only the single best-scoring
      cell. Silently drops the rest of a multi-part region (box too small / off to one
      side) -- kept here only to reproduce the original bug for A/B comparison.

    blob is computed on the coarse feature grid (e.g. 32x32), not image pixels: a single
    anatomical region often splits into several grid-cells-wide CCs that only look merged
    after upsampling for display."""
    from skimage.measure import label as cc_label

    h, w = score.shape
    H, W = img_hw

    if cc_mode == 'seed_only':
        lab = cc_label(blob)
        seed = np.unravel_index(np.argmax(np.where(blob, score, -np.inf)), score.shape)
        sel = (lab == lab[seed]) if lab[seed] else blob
    elif cc_mode == 'union':
        lab = cc_label(blob)
        n = lab.max()
        sizes = np.bincount(lab.ravel())
        sel = np.isin(lab, [i for i in range(1, n + 1) if sizes[i] >= min_cc_px])
        if not sel.any():
            sel = blob
    elif cc_mode == 'dilate_largest':
        from scipy.ndimage import binary_dilation
        grown = binary_dilation(blob, iterations=dilate_iters) if dilate_iters > 0 else blob
        lab = cc_label(grown)
        if lab.max() == 0:
            sel = blob
        else:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            sel = (lab == sizes.argmax()) & blob
            if not sel.any():
                sel = blob
    else:
        raise ValueError(f'unknown cc_mode: {cc_mode!r}')

    ys, xs = np.where(sel)
    x0, x1 = float(xs.min()) / w * W, float(xs.max() + 1) / w * W
    y0, y1 = float(ys.min()) / h * H, float(ys.max() + 1) / h * H
    return (max(0.0, x0 - margin_px), max(0.0, y0 - margin_px),
            min(float(W), x1 + margin_px), min(float(H), y1 + margin_px))


def _mask_from_blob(blob: np.ndarray, score: np.ndarray, cell_px: float,
                    dilate_iters: int = 1, cc_mode: str = 'dilate_largest',
                    min_cc_px: int = 2, min_frac: float = 0.3,
                    min_abs_cells: float = 2.0,
                    coherence_frac: float | None = None) -> np.ndarray:
    """blob/score are already at FULL image resolution (winner-take-all decided on
    bilinear-upsampled score maps -- same math as debug_medsam2.py's winner-map
    visualization, which is why that panel already looks smooth). This just picks which
    pixels of the blob feed the mask prompt (same cc_mode semantics as _box_from_blob),
    scaled from grid-cell units to pixel units via cell_px (~coarse-grid cell size in
    full-res pixels). Deliberately a standalone copy of _box_from_blob's cell-selection
    logic (not a shared refactor) so the box-prompt path stays byte-for-byte untouched --
    revert = delete this + multiclass_masks + segment_volume_mask.

    Earlier version did selection on the coarse grid then nearest-upsampled the binary
    result, which gave blocky/fragmented ("bbox-like") masks -- see git history.

    Pixel-precise winner-take-all sometimes lets `sel` (the chosen connected component)
    end up a tiny fragment of `blob` (the full winning region) -- SAM2 then has little to
    segment from (dice=0). A fixed pixel/cell floor here would misfire on anatomy that is
    just small in absolute terms (e.g. a kidney cross-section) but whose `sel` already
    covers all of `blob` cleanly -- that case needs no fallback. So the floor is relative:
    fall back to the filled bounding box of the whole blob only when `sel` covers less than
    `min_frac` of `blob`'s own area (real fragmentation), not based on absolute size. A
    tiny `min_abs_cells` pixel floor still catches the degenerate case where `blob` itself
    is just noise (1-2 px).

    coherence_frac (None = old behaviour, exact): when the fallback fires, `blob` itself can
    be scattered noise spread over a big chunk of the frame (weak/noisy class, score_thresh
    ~0 lets almost any positive-margin pixel in) -- filling its raw bbox then balloons the
    prompt across anatomy that isn't the target class at all. When set, the fallback first
    drops `blob` pixels farther than `coherence_frac * diag(blob's own bbox)` from `blob`'s
    own centroid (self-referential -- no cross-frame info needed here), then bboxes what's
    left. Same fraction for every class/dataset -- not a per-muscle tuning knob, just "ignore
    outlier pixels far from this blob's own center of mass" before trusting its extent."""
    from skimage.measure import label as cc_label

    dilate_px = max(1, round(dilate_iters * cell_px))
    min_px = max(1, round(min_cc_px * cell_px * cell_px))

    if cc_mode == 'seed_only':
        lab = cc_label(blob)
        seed = np.unravel_index(np.argmax(np.where(blob, score, -np.inf)), score.shape)
        sel = (lab == lab[seed]) if lab[seed] else blob
    elif cc_mode == 'union':
        lab = cc_label(blob)
        n = lab.max()
        sizes = np.bincount(lab.ravel())
        sel = np.isin(lab, [i for i in range(1, n + 1) if sizes[i] >= min_px])
        if not sel.any():
            sel = blob
    elif cc_mode == 'dilate_largest':
        from scipy.ndimage import binary_dilation
        grown = binary_dilation(blob, iterations=dilate_px) if dilate_px > 0 else blob
        lab = cc_label(grown)
        if lab.max() == 0:
            sel = blob
        else:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            sel = (lab == sizes.argmax()) & blob
            if not sel.any():
                sel = blob
    else:
        raise ValueError(f'unknown cc_mode: {cc_mode!r}')

    min_abs_px = min_abs_cells * cell_px * cell_px
    min_area_px = max(min_frac * blob.sum(), min_abs_px)
    if sel.sum() < min_area_px:
        fallback_blob = blob
        if coherence_frac is not None:
            ys, xs = np.where(blob)
            cy, cx = ys.mean(), xs.mean()
            diag = float(np.hypot(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1))
            near = np.hypot(ys - cy, xs - cx) <= coherence_frac * diag
            if near.any():
                fallback_blob = np.zeros_like(blob)
                fallback_blob[ys[near], xs[near]] = True
        ys, xs = np.where(fallback_blob)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        sel = np.zeros_like(blob)
        sel[y0:y1, x0:x1] = True

    return sel


def multiclass_masks(score_maps: dict, query_body2d: np.ndarray, img_hw: tuple,
                     left_is_low_x: bool | None = None, score_thresh: float = 0.0,
                     single_leg: bool = False, cc_mode: str = 'dilate_largest',
                     coherence_frac: float | None = None) -> dict:
    """Mask-prompt sibling of multiclass_boxes: same winner-take-all competition, but
    decided on bilinear-upsampled full-res score maps (matching debug_medsam2.py's
    winner-map visualization) and returned as a full-res boolean mask (pseudo-label)
    instead of reduced to an axis-aligned box. Separate function (not a flag on
    multiclass_boxes) so the box-oracle path is untouched -- revert = delete this +
    _mask_from_blob + segment_volume_mask.

    Returns {'<side>_<type>': (score, mask_HxW_bool)}, or {'<type>': ...} when single_leg.
    """
    names = sorted(score_maps)
    h, w = score_maps[names[0]].shape
    H, W = img_hw
    cell_px = ((H / h) + (W / w)) / 2.0

    stack = np.stack([score_maps[c] for c in names])   # [K,h,w]
    t = torch.from_numpy(stack.astype(np.float32))[None]
    ups = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0].numpy()  # [K,H,W]
    win, best = ups.argmax(0), ups.max(0)

    sides = {'': _largest_cc(query_body2d)} if single_leg else side_masks(query_body2d, left_is_low_x)

    out = {}
    for si, smask in sides.items():
        leg = smask > 0.5   # already full-res -- no coarse-grid downsample needed
        for ci, c in enumerate(names):
            blob = (win == ci) & (best > score_thresh) & leg
            if not blob.any():
                continue
            key = c if single_leg else f'{si}_{c}'
            mask = _mask_from_blob(blob, best, cell_px, cc_mode=cc_mode,
                                   coherence_frac=coherence_frac)
            out[key] = (float(best[blob].max()), mask)
    return out


def multiclass_masks_for_frame(seg, bags: dict, frame_u8: np.ndarray,
                               left_is_low_x: bool | None = None,
                               body_thresh: float = 10.0, body_min_px: int = 50,
                               score_thresh: float = 0.0, single_leg: bool = False,
                               cc_mode: str = 'dilate_largest',
                               coherence_frac: float | None = None) -> tuple:
    """Mask-prompt sibling of multiclass_boxes_for_frame. One query frame -> all masks
    at once. Returns ({'<side>_<type>': (score, mask)} or {'<type>': (score, mask)}
    when single_leg, score_maps)."""
    feat = seg.embed_frame(frame_u8)
    score_maps = multiclass_score_maps(feat, bags)
    body = body_mask2d(frame_u8, body_thresh, body_min_px)
    masks = multiclass_masks(score_maps, body, frame_u8.shape, left_is_low_x,
                             score_thresh, single_leg=single_leg, cc_mode=cc_mode,
                             coherence_frac=coherence_frac)
    return masks, score_maps


def multiclass_mask_anchors(seg, bags: dict, cand_frames: list, label_name: str,
                            left_is_low_x_val: bool | None = None,
                            n_anchors: int = 1, min_gap: int = 3,
                            body_thresh: float = 10.0, body_min_px: int = 50,
                            score_thresh: float = 0.0, single_leg: bool = False,
                            cc_mode: str = 'dilate_largest',
                            coherence_frac: float | None = None,
                            key_fidx: int | None = None) -> dict:
    """Mask-prompt sibling of support_anchors_dense_bodymasked_bbox, for ONE class.

    Fixes two things the frozen-single-key-slice call in eval_medsam2.py cannot:

    - Catastrophic zero. The winner-take-all in multiclass_masks is per cell, so on an
      unlucky slice a rival type can take every cell of the true region and the class
      simply disappears from the returned dict -- eval then scores the WHOLE volume as
      empty (Dice 0), not just that slice. Measured on the mcvis_label runs: 22 scans at
      Dice ~0 and 49 below 0.2 across the 6 datasets. Scoring several candidate slices and
      keeping the ones where the class actually survives turns most of those into a normal
      propagation instead of a zero.
    - Single anchor. segment_volume_mask already conditions on every prompted frame, but
      only one was ever passed. Prompting the N best surviving slices (spread along z by
      the same greedy as pick_anchors) restarts propagation before the mask decays, the
      same reason --n_anchors helps the box path (B4).

    cand_frames: [(local_frame_idx, frame_u8)] candidates, already cropped/indexed like
    vol_u8. n_anchors=1 with a single candidate reproduces the previous single-key-slice
    call exactly (same lookup, no extra encoder passes).

    coherence_frac (None = old behaviour, exact): pick_anchors alone ranks candidate slices
    purely by peak winner-take-all score, with zero check that the winning blob sits in the
    same place as on other slices. A slice where a stray high-margin cluster wins in the
    WRONG spot (rival muscle's territory, a bone/fascia edge -- common for a small/weak-
    texture class near score_thresh~0) can outrank the slice where the class won correctly
    but with a weaker margin, and gets accepted as an anchor. segment_volume_mask then
    conditions SAM2's propagation on that wrong blob together with the correct ones, which
    merges or drags the whole-volume prediction toward the wrong location -- worse than
    plain single-anchor loss because it actively corrupts otherwise-good slices, not just
    misses one. When set, candidates are filtered by centroid distance to a reference point
    BEFORE pick_anchors runs: the key slice's own blob centroid if the class survives there
    (key_fidx), else the component-wise median centroid across all candidates (robust to a
    minority of outliers). Distance threshold is `coherence_frac * diag(frame)` -- one
    fraction, same for every class/dataset (no per-muscle tuning): slices near each other in
    z show a muscle in roughly the same place, so a candidate whose blob teleported away
    from where the class otherwise sits is discarded before it can become an anchor.
    Also forwarded into multiclass_masks_for_frame -> _mask_from_blob, so its scattered-blob
    fallback (see _mask_from_blob docstring) gets the same self-referential fix while
    building these very candidates.

    returns: {local_frame_idx -> mask_HxW_bool}, ready for
    MedSAM2Segmenter.segment_volume_mask. Empty dict = the class lost on every candidate
    (caller falls back to an empty volume, as before).
    """
    cands = []
    for fidx, frame_u8 in cand_frames:
        masks_by_name, _ = multiclass_masks_for_frame(
            seg, bags, frame_u8, left_is_low_x_val,
            body_thresh=body_thresh, body_min_px=body_min_px,
            score_thresh=score_thresh, single_leg=single_leg,
            cc_mode=cc_mode, coherence_frac=coherence_frac)
        if label_name in masks_by_name:
            score, mask = masks_by_name[label_name]
            cands.append((score, fidx, mask))

    if coherence_frac is not None and len(cands) > 1:
        centroids = {}
        for _, fidx, mask in cands:
            ys, xs = np.where(mask)
            centroids[fidx] = (ys.mean(), xs.mean())
        if key_fidx is not None and key_fidx in centroids:
            ref = centroids[key_fidx]
        else:
            all_c = np.array(list(centroids.values()))
            ref = (float(np.median(all_c[:, 0])), float(np.median(all_c[:, 1])))
        H, W = cand_frames[0][1].shape
        diag = float(np.hypot(H, W))
        cands = [c for c in cands
                if np.hypot(centroids[c[1]][0] - ref[0],
                            centroids[c[1]][1] - ref[1]) <= coherence_frac * diag]

    return {fidx: mask for _, fidx, mask in pick_anchors(cands, n_anchors, min_gap)}


def _largest_cc(mask2d: np.ndarray) -> np.ndarray:
    """Largest connected component of mask2d. single_leg datasets: the raw query frame's
    FOV sometimes still shows BOTH legs even though only one is annotated -- without this,
    the single_leg group in multiclass_boxes is the whole frame, so a type winning even a
    few noise cells on the other, unannotated leg drags the box all the way across the
    midline to include it."""
    from skimage.measure import label as cc_label

    lab = cc_label(mask2d)
    if lab.max() == 0:
        return mask2d
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return lab == sizes.argmax()


def _neg_points_from_rivals(win: np.ndarray, best: np.ndarray, ci: int, blob: np.ndarray,
                            leg: np.ndarray, img_hw: tuple, score_thresh: float,
                            max_points: int = 3) -> list:
    """Cells inside blob's own bbox that a RIVAL type won -- exactly the neighboring
    tissue an elongated box (e.g. soleus spanning the calf) pulls in alongside the real
    muscle. Returned as negative-click points in image (x,y) coords, top max_points by
    rival score, for segment_volume's box+neg-points hybrid (prompt_mode=support_multiclass
    --neg_points)."""
    h, w = best.shape
    H, W = img_hw
    ys, xs = np.where(blob)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    rival = np.zeros_like(blob)
    rival[y0:y1, x0:x1] = ((win[y0:y1, x0:x1] != ci) & (best[y0:y1, x0:x1] > score_thresh)
                           & leg[y0:y1, x0:x1])
    ry, rx = np.where(rival)
    if len(ry) == 0:
        return []

    order = np.argsort(-best[ry, rx])[:max_points]
    return [((float(rx[i]) + 0.5) / w * W, (float(ry[i]) + 0.5) / h * H) for i in order]


def multiclass_boxes(score_maps: dict, query_body2d: np.ndarray, img_hw: tuple,
                     left_is_low_x: bool | None = None, score_thresh: float = 0.0,
                     margin_px: float = 0.0, single_leg: bool = False,
                     cc_mode: str = 'dilate_largest', neg_points: bool = False,
                     max_neg_points: int = 3) -> dict:
    """Winner-take-all per cell across the types, then one box per (side, type): the cells
    that type c wins, intersected with that leg. Two types can no longer claim the same
    pixels, and a confident false match on bone marrow dies because BG wins there.
    Returns {'<side>_<type>': (score, box_xyxy)}, or {'<side>_<type>': (score, box_xyxy,
    neg_pts)} when neg_points=True -- neg_pts are rival-won cells inside the box, meant as
    negative clicks alongside the box to push SAM2 off neighboring muscle (elongated
    shapes like soleus otherwise drag a big axis-aligned box across the neighbor).

    single_leg=True: one leg per volume (no L/R split) -- the whole body mask is used as
    the single group and keys are bare type names (no side prefix)."""
    names = sorted(score_maps)
    stack = np.stack([score_maps[c] for c in names])   # [K,h,w]
    win, best = stack.argmax(0), stack.max(0)
    h, w = best.shape

    sides = {'': _largest_cc(query_body2d)} if single_leg else side_masks(query_body2d, left_is_low_x)

    out = {}
    for si, smask in sides.items():
        leg = _to_grid(smask, h, w).numpy() > 0.5
        for ci, c in enumerate(names):
            blob = (win == ci) & (best > score_thresh) & leg
            if not blob.any():
                continue
            key = c if single_leg else f'{si}_{c}'
            box = _box_from_blob(blob, best, img_hw, margin_px, cc_mode=cc_mode)
            if neg_points:
                pts = _neg_points_from_rivals(win, best, ci, blob, leg, img_hw,
                                              score_thresh, max_neg_points)
                out[key] = (float(best[blob].max()), box, pts)
            else:
                out[key] = (float(best[blob].max()), box)
    return out


def multiclass_boxes_for_frame(seg, bags: dict, frame_u8: np.ndarray,
                               left_is_low_x: bool | None = None,
                               body_thresh: float = 10.0, body_min_px: int = 50,
                               score_thresh: float = 0.0, margin_px: float = 0.0,
                               single_leg: bool = False,
                               cc_mode: str = 'dilate_largest', neg_points: bool = False,
                               max_neg_points: int = 3) -> tuple:
    """One query frame -> all boxes at once. returns ({'<side>_<type>': (score, box)}
    or {'<type>': (score, box)} when single_leg, score_maps) -- the maps come back for
    the debug overlay. neg_points=True adds a third tuple element per key (see
    multiclass_boxes)."""
    feat = seg.embed_frame(frame_u8)
    score_maps = multiclass_score_maps(feat, bags)
    body = body_mask2d(frame_u8, body_thresh, body_min_px)
    boxes = multiclass_boxes(score_maps, body, frame_u8.shape, left_is_low_x,
                             score_thresh, margin_px, single_leg=single_leg, cc_mode=cc_mode,
                             neg_points=neg_points, max_neg_points=max_neg_points)
    return boxes, score_maps


# ============================ B2 support: bilateral bookkeeping ============================
# Moved here from scripts/eval/debug_medsam2.py (cmd_mcvis) so both the debug-vis tool and
# a full propagation-based eval (eval_medsam2.py, prompt_mode=support_multiclass) share one
# implementation instead of two copies drifting apart.


def muscle_types(label_names: list) -> dict:
    """{'QF': {'L': 1, 'R': 5}, ...} -- the L and R label ids of each muscle type.
    Requires the '<side>_<type>' naming convention; types missing one side are dropped."""
    types = defaultdict(dict)
    for lv, name in enumerate(label_names):
        if lv == 0 or '_' not in name:
            continue
        side, mtype = name.split('_', 1)
        types[mtype][side] = lv
    return {t: v for t, v in types.items() if len(v) == 2}


def muscle_types_single(label_names: list) -> dict:
    """{'QF': 1, ...} -- single-leg datasets, one label id per type (no L/R to pool)."""
    return {name: lv for lv, name in enumerate(label_names) if lv != 0}


def left_is_low_x(lbl: np.ndarray, types: dict) -> bool:
    """Scanner side convention, read off the support GT (never the query)."""
    lx = [np.where(lbl == v['L'])[2].mean() for v in types.values() if (lbl == v['L']).any()]
    rx = [np.where(lbl == v['R'])[2].mean() for v in types.values() if (lbl == v['R']).any()]
    return mean(lx) < mean(rx)


def support_bag_slices(supp_vol_u8: np.ndarray, supp_lbl: np.ndarray, types: dict,
                       k: int, min_gap: int) -> tuple:
    """Union over types of their K best slices; each kept slice carries every type's mask
    (L+R pooled). One embed per slice, all bags filled from it. Returns (slices, zs) where
    slices = [(frame_u8, {cls: mask2d})], ready for build_multiclass_bags."""
    zs = set()
    for v in types.values():
        fg = ((supp_lbl == v['L']) | (supp_lbl == v['R'])).astype(np.uint8)
        if fg.any():
            zs |= set(pick_support_slices(fg, k, min_gap))

    out = []
    for z in sorted(zs):
        masks = {t: ((supp_lbl[z] == v['L']) | (supp_lbl[z] == v['R']))
                 for t, v in types.items()}
        out.append((supp_vol_u8[z], {t: m for t, m in masks.items() if m.any()}))
    return out, sorted(zs)


def support_bag_slices_single(supp_vol_u8: np.ndarray, supp_lbl: np.ndarray, types: dict,
                              k: int, min_gap: int) -> tuple:
    """Single-leg version of support_bag_slices: one label id per type, no L+R pooling."""
    zs = set()
    for lv in types.values():
        fg = (supp_lbl == lv).astype(np.uint8)
        if fg.any():
            zs |= set(pick_support_slices(fg, k, min_gap))

    out = []
    for z in sorted(zs):
        masks = {t: (supp_lbl[z] == lv) for t, lv in types.items()}
        out.append((supp_vol_u8[z], {t: m for t, m in masks.items() if m.any()}))
    return out, sorted(zs)
