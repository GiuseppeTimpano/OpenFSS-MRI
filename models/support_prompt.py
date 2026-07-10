"""
PerSAM-style one-shot BOX prompting from a support volume+mask, using SAM2's own image
encoder. Used by scripts/eval/eval_medsam2.py (prompt_mode=support_bbox).

Point-prompt variants were removed once box prompting superseded them (R_HS, 20 HV scans:
point Dice 0.3673 vs box 0.6370). See git history.
"""
from collections import defaultdict

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


def _box_from_blob(blob: np.ndarray, score: np.ndarray, img_hw: tuple,
                   margin_px: float = 0.0, dilate_iters: int = 1) -> tuple:
    """Largest connected component of blob (dilated by dilate_iters cells first, to merge
    an anatomical region's own separate coarse-grid pieces -- e.g. HS's 3 heads, sitting a
    cell or two apart at this resolution -- without a distant noise speck a few cells away
    also merging in) -> box in (H,W) px.

    blob is computed on the coarse feature grid (e.g. 32x32), not image pixels: a single
    anatomical region often splits into several grid-cells-wide CCs that only look merged
    after upsampling for display. The old single-seed-CC selection then silently dropped
    the rest of the true region (box too small / off to one side); the naive "keep every
    CC" fix over-corrected -- a lone stray winning cell anywhere in the frame, however far
    from the real region, then ballooned the box to include it (e.g. QF_06: a couple of
    QF-won cells near the AD region dragged the box across half the leg). Dilating first
    merges only genuinely nearby pieces; the box itself is still cropped back to the real
    (undilated) blob pixels, so dilation never inflates it directly."""
    from scipy.ndimage import binary_dilation
    from skimage.measure import label as cc_label

    h, w = score.shape
    H, W = img_hw
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

    ys, xs = np.where(sel)
    x0, x1 = float(xs.min()) / w * W, float(xs.max() + 1) / w * W
    y0, y1 = float(ys.min()) / h * H, float(ys.max() + 1) / h * H
    return (max(0.0, x0 - margin_px), max(0.0, y0 - margin_px),
            min(float(W), x1 + margin_px), min(float(H), y1 + margin_px))


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


def multiclass_boxes(score_maps: dict, query_body2d: np.ndarray, img_hw: tuple,
                     left_is_low_x: bool | None = None, score_thresh: float = 0.0,
                     margin_px: float = 0.0, single_leg: bool = False) -> dict:
    """Winner-take-all per cell across the types, then one box per (side, type): the cells
    that type c wins, intersected with that leg. Two types can no longer claim the same
    pixels, and a confident false match on bone marrow dies because BG wins there.
    Returns {'<side>_<type>': (score, box_xyxy)}.

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
            out[key] = (float(best[blob].max()),
                       _box_from_blob(blob, best, img_hw, margin_px))
    return out


def multiclass_boxes_for_frame(seg, bags: dict, frame_u8: np.ndarray,
                               left_is_low_x: bool | None = None,
                               body_thresh: float = 10.0, body_min_px: int = 50,
                               score_thresh: float = 0.0, margin_px: float = 0.0,
                               single_leg: bool = False) -> tuple:
    """One query frame -> all boxes at once. returns ({'<side>_<type>': (score, box)}
    or {'<type>': (score, box)} when single_leg, score_maps) -- the maps come back for
    the debug overlay."""
    feat = seg.embed_frame(frame_u8)
    score_maps = multiclass_score_maps(feat, bags)
    body = body_mask2d(frame_u8, body_thresh, body_min_px)
    boxes = multiclass_boxes(score_maps, body, frame_u8.shape, left_is_low_x,
                             score_thresh, margin_px, single_leg=single_leg)
    return boxes, score_maps
