"""
PerSAM-style one-shot BOX prompting from a support volume+mask, using SAM2's own image
encoder. Used by scripts/eval/eval_medsam2.py (prompt_mode=support_bbox).

Point-prompt variants were removed once box prompting superseded them (R_HS, 20 HV scans:
point Dice 0.3673 vs box 0.6370). See git history.
"""
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
    the whole negative bag (nearest-neighbor matching). Returns (pos_map, neg_map) [h,w];
    an empty bag yields an all -1 map."""
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
