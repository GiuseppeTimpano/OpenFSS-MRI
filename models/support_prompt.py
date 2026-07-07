"""
PerSAM-style one-shot point prompting from a support volume+mask, using SAM2's own
image encoder (no external/trained network, no cross-patient registration -- see
~/.claude/plans/peppy-fluttering-waffle.md and HANDOFF.md for rationale).

Used by scripts/eval/eval_medsam2.py (prompt_mode=support) together with
models/medsam2_adapter.py's MedSAM2Segmenter.embed_frame / segment_volume_points.
"""
import numpy as np
import torch
import torch.nn.functional as F


def key_slice(fg_mask: np.ndarray) -> int:
    """fg_mask: [Z,H,W] bool/0-1. Returns z-index of the max-FG-area slice."""
    areas = fg_mask.reshape(fg_mask.shape[0], -1).sum(axis=1)
    return int(np.argmax(areas))


def masked_prototype(feat: torch.Tensor, mask2d: np.ndarray) -> torch.Tensor:
    """feat: [C,h,w] encoder feature. mask2d: [H,W] bool/0-1 at original resolution.
    Returns [C] masked-average-pooled prototype (mask resized to feat's h,w grid)."""
    C, h, w = feat.shape
    m = torch.from_numpy(mask2d.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    m = m.to(feat.device)
    denom = m.sum().clamp_min(1e-6)
    return (feat * m[None]).sum(dim=(1, 2)) / denom


def similarity_map(feat: torch.Tensor, proto: torch.Tensor) -> np.ndarray:
    """feat: [C,h,w], proto: [C]. Returns [h,w] cosine similarity map."""
    f = F.normalize(feat, dim=0)
    p = F.normalize(proto, dim=0)
    sim = (f * p[:, None, None]).sum(dim=0)
    return sim.cpu().numpy()


def pick_points(sim_map: np.ndarray, img_hw: tuple) -> tuple:
    """sim_map: [h,w] similarity. img_hw: (H,W) of the target frame.
    Returns (pos_xy, neg_xy) pixel coords in img_hw (argmax/argmin, patch center)."""
    h, w = sim_map.shape
    H, W = img_hw
    pos_idx = np.unravel_index(np.argmax(sim_map), sim_map.shape)
    neg_idx = np.unravel_index(np.argmin(sim_map), sim_map.shape)

    def to_xy(idx):
        py, px = idx
        return (float((px + 0.5) / w * W), float((py + 0.5) / h * H))

    return to_xy(pos_idx), to_xy(neg_idx)


def support_prompt_for_query(seg, supp_frame_u8: np.ndarray, supp_mask2d: np.ndarray,
                              query_frames: list) -> tuple:
    """
    seg           : MedSAM2Segmenter (uses seg.embed_frame; no other coupling).
    supp_frame_u8 : [H,W] uint8, support key slice (max-FG-area, via key_slice).
    supp_mask2d   : [H,W] bool/0-1, support GT mask on that slice.
    query_frames  : list[(frame_idx, frame_u8[H,W])] candidate query frames.

    Picks the query frame with the highest max-similarity to the support prototype
    (no query GT read anywhere) and returns its point prompt.

    returns: (frame_idx, pos_xy, neg_xy)
    """
    supp_feat = seg.embed_frame(supp_frame_u8)
    proto = masked_prototype(supp_feat, supp_mask2d)

    best = None  # (score, frame_idx, pos_xy, neg_xy)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        sim = similarity_map(feat, proto)
        pos_xy, neg_xy = pick_points(sim, frame_u8.shape)
        score = float(sim.max())
        if best is None or score > best[0]:
            best = (score, fidx, pos_xy, neg_xy)
    return best[1], best[2], best[3]


def extract_support_vectors(feat: torch.Tensor, mask2d: np.ndarray,
                             thr_hi: float = 0.7, thr_lo: float = 0.3) -> tuple:
    """feat: [C,h,w] encoder feature. mask2d: [H,W] bool/0-1 at original resolution.
    Resizes mask to (h,w) and splits cells into a foreground bag (mask_r > thr_hi) and
    a background bag (mask_r < thr_lo), dropping the ambiguous boundary band in between
    (bilinear-downsampled edge cells) instead of collapsing everything into one mean
    vector -- see discussion on masked_prototype's information loss.
    Returns (Pos_n [C,Np], Neg_n [C,Nn]), L2-normalized along C. Np or Nn may be 0
    (e.g. a support mask small enough that no cell clears thr_hi)."""
    C, h, w = feat.shape
    m = torch.from_numpy(mask2d.astype(np.float32))[None, None]
    m = F.interpolate(m, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    m = m.to(feat.device)

    feat_flat = feat.reshape(C, h * w)
    mask_flat = m.reshape(h * w)

    pos_idx = (mask_flat > thr_hi).nonzero(as_tuple=True)[0]
    neg_idx = (mask_flat < thr_lo).nonzero(as_tuple=True)[0]

    Pos = feat_flat[:, pos_idx]
    Neg = feat_flat[:, neg_idx]
    Pos_n = F.normalize(Pos, dim=0) if Pos.shape[1] > 0 else Pos
    Neg_n = F.normalize(Neg, dim=0) if Neg.shape[1] > 0 else Neg
    return Pos_n, Neg_n


def dense_similarity_maps(feat_query: torch.Tensor, Pos_n: torch.Tensor,
                           Neg_n: torch.Tensor) -> tuple:
    """feat_query: [C,h,w]. Pos_n/Neg_n: [C,Np]/[C,Nn] from extract_support_vectors.
    For every query spatial cell, takes the max cosine similarity against the whole
    positive bag and, separately, the whole negative bag (nearest-neighbor matching,
    not a dot product with a single averaged prototype). Returns (pos_map, neg_map),
    each [h,w] numpy; an empty bag yields an all -1 map (no match possible)."""
    C, h, w = feat_query.shape
    Q_n = F.normalize(feat_query.reshape(C, h * w), dim=0)  # [C,M]

    def _max_sim(Bag_n):
        if Bag_n.shape[1] == 0:
            return -np.ones((h, w), dtype=np.float32)
        sim = Bag_n.t() @ Q_n                 # [N,M]
        return sim.max(dim=0).values.reshape(h, w).cpu().numpy()

    return _max_sim(Pos_n), _max_sim(Neg_n)


def pick_points_dense(pos_map: np.ndarray, neg_map: np.ndarray, img_hw: tuple) -> tuple:
    """pos_map/neg_map: [h,w] (from dense_similarity_maps). img_hw: (H,W) of the target
    frame. Positive point = argmax(pos_map - neg_map) (FG-like AND not BG-like);
    negative point = argmax(neg_map) (explicit best match to the support's background,
    not merely 'least like the object'). Returns (pos_xy, neg_xy) in img_hw pixel coords."""
    h, w = pos_map.shape
    H, W = img_hw

    def to_xy(idx):
        py, px = idx
        return (float((px + 0.5) / w * W), float((py + 0.5) / h * H))

    score_map = pos_map - neg_map
    pos_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
    neg_idx = np.unravel_index(np.argmax(neg_map), neg_map.shape)
    return to_xy(pos_idx), to_xy(neg_idx)


def support_prompt_for_query_dense(seg, supp_frame_u8: np.ndarray, supp_mask2d: np.ndarray,
                                    query_frames: list, thr_hi: float = 0.7,
                                    thr_lo: float = 0.3) -> tuple:
    """Dense-matching counterpart of support_prompt_for_query: uses
    extract_support_vectors + dense_similarity_maps + pick_points_dense instead of a
    single masked_prototype + similarity_map + pick_points. Same call signature/return
    shape as support_prompt_for_query, added alongside it (not a replacement).

    returns: (frame_idx, pos_xy, neg_xy)
    """
    supp_feat = seg.embed_frame(supp_frame_u8)
    Pos_n, Neg_n = extract_support_vectors(supp_feat, supp_mask2d, thr_hi, thr_lo)

    best = None  # (score, frame_idx, pos_xy, neg_xy)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        pos_xy, neg_xy = pick_points_dense(pos_map, neg_map, frame_u8.shape)
        score = float((pos_map - neg_map).max())
        if best is None or score > best[0]:
            best = (score, fidx, pos_xy, neg_xy)
    return best[1], best[2], best[3]


def body_mask2d(img_u8: np.ndarray, thresh: float = 10.0,
                 min_component_px: int = 50) -> np.ndarray:
    """img_u8: [H,W] uint8 [0,255]. Thresholds air/padding out, keeps EVERY connected
    component >= min_component_px (not just the single largest -- bilateral anatomy,
    e.g. left/right thigh, is two separate blobs of comparable size; keeping only the
    largest silently drops the other leg, verified on MRI_muscle). Fills internal holes
    per surviving region. Returns [H,W] bool body mask."""
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
    """Like extract_support_vectors, but the background bag (Neg_n) is further
    restricted to cells inside the support's body mask (supp_body2d: [H,W] bool, from
    body_mask2d) -- excludes air/padding vectors from the support's background bag."""
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


def pick_points_dense_bodymasked(pos_map: np.ndarray, neg_map: np.ndarray,
                                  query_body2d: np.ndarray, img_hw: tuple) -> tuple:
    """Like pick_points_dense, but the negative point search is restricted to cells
    inside the query's body mask (query_body2d: [H,W] bool, from body_mask2d) -- avoids
    picking a negative point in air/padding outside the patient's body (empirically the
    global argmax(neg_map) tends to land there: air-vs-air similarity saturates near 1.0
    since both the support's background bag and the query frame contain large, uniform
    air regions). The positive point (argmax(pos_map - neg_map), on the ORIGINAL,
    unmasked neg_map) is unchanged."""
    h, w = pos_map.shape
    H, W = img_hw

    def to_xy(idx):
        py, px = idx
        return (float((px + 0.5) / w * W), float((py + 0.5) / h * H))

    score_map = pos_map - neg_map
    pos_idx = np.unravel_index(np.argmax(score_map), score_map.shape)

    body = torch.from_numpy(query_body2d.astype(np.float32))[None, None]
    body = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    body_grid = (body.numpy() > 0.5)
    neg_map_masked = np.where(body_grid, neg_map, -1.0)
    neg_idx = np.unravel_index(np.argmax(neg_map_masked), neg_map_masked.shape)

    return to_xy(pos_idx), to_xy(neg_idx)


def support_prompt_for_query_dense_bodymasked(seg, supp_frame_u8: np.ndarray,
                                               supp_mask2d: np.ndarray, query_frames: list,
                                               thr_hi: float = 0.7, thr_lo: float = 0.3,
                                               body_thresh: float = 10.0,
                                               body_min_px: int = 50) -> tuple:
    """Body-mask-aware counterpart of support_prompt_for_query_dense: restricts both the
    support's background bag and each query frame's negative-point search to inside the
    patient's body (body_mask2d), instead of allowing negatives in air/padding. Same call
    signature/return shape, added alongside support_prompt_for_query_dense (not a
    replacement).

    returns: (frame_idx, pos_xy, neg_xy)
    """
    supp_feat = seg.embed_frame(supp_frame_u8)
    supp_body = body_mask2d(supp_frame_u8, body_thresh, body_min_px)
    Pos_n, Neg_n = extract_support_vectors_bodymasked(supp_feat, supp_mask2d, supp_body,
                                                       thr_hi, thr_lo)

    best = None  # (score, frame_idx, pos_xy, neg_xy)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, body_thresh, body_min_px)
        pos_xy, neg_xy = pick_points_dense_bodymasked(pos_map, neg_map, q_body, frame_u8.shape)
        score = float((pos_map - neg_map).max())
        if best is None or score > best[0]:
            best = (score, fidx, pos_xy, neg_xy)
    return best[1], best[2], best[3]


def mask_bbox_diag(mask2d: np.ndarray) -> float:
    """mask2d: [H,W] bool/0-1. Bbox diagonal in pixels, 0.0 if mask empty. Used as a
    muscle-scale proxy (support GT bbox, no query GT read) for the local negative
    search window in pick_points_dense_bodymasked_local."""
    ys, xs = np.where(mask2d)
    if ys.size == 0:
        return 0.0
    return float(np.hypot(ys.max() - ys.min(), xs.max() - xs.min()))


def pick_points_dense_bodymasked_local(pos_map: np.ndarray, neg_map: np.ndarray,
                                        query_body2d: np.ndarray, img_hw: tuple,
                                        radius_px: float) -> tuple:
    """Like pick_points_dense_bodymasked, but the negative point search is further
    restricted to a local window of radius_px pixels around the chosen positive point
    (still intersected with the body mask) -- a body-masked global argmax can still land
    very far from the target (e.g. the contralateral leg), which is a weak negative for
    SAM2 (negatives are most useful carving the boundary against a NEARBY confusor).
    Falls back to the full body mask (no window) if the local window contains no
    in-body cells (e.g. positive point near the body edge with a small radius) --
    same behavior as pick_points_dense_bodymasked in that case. The positive point
    (argmax(pos_map - neg_map), on the ORIGINAL unmasked neg_map) is unchanged."""
    h, w = pos_map.shape
    H, W = img_hw

    def to_xy(idx):
        py, px = idx
        return (float((px + 0.5) / w * W), float((py + 0.5) / h * H))

    score_map = pos_map - neg_map
    pos_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
    pos_xy = to_xy(pos_idx)

    body = torch.from_numpy(query_body2d.astype(np.float32))[None, None]
    body = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    body_grid = (body.numpy() > 0.5)

    yy, xx = np.mgrid[0:h, 0:w]
    cell_cx = (xx + 0.5) / w * W
    cell_cy = (yy + 0.5) / h * H
    dist = np.hypot(cell_cx - pos_xy[0], cell_cy - pos_xy[1])
    window = dist <= radius_px

    candidates = body_grid & window
    if not candidates.any():
        candidates = body_grid
    neg_map_masked = np.where(candidates, neg_map, -1.0)
    neg_idx = np.unravel_index(np.argmax(neg_map_masked), neg_map_masked.shape)

    return pos_xy, to_xy(neg_idx)


def support_prompt_for_query_dense_bodymasked_local(seg, supp_frame_u8: np.ndarray,
                                                     supp_mask2d: np.ndarray, query_frames: list,
                                                     thr_hi: float = 0.7, thr_lo: float = 0.3,
                                                     body_thresh: float = 10.0,
                                                     body_min_px: int = 50,
                                                     radius_k: float = 1.5) -> tuple:
    """Local-window counterpart of support_prompt_for_query_dense_bodymasked: same
    body-masked positive/background bags, but the negative point search per query frame
    is further restricted to a window around the positive point, radius_px = radius_k *
    mask_bbox_diag(supp_mask2d) (support GT bbox as a muscle-scale proxy -- no query GT
    read). Same call signature/return shape, added alongside
    support_prompt_for_query_dense_bodymasked (not a replacement).

    returns: (frame_idx, pos_xy, neg_xy)
    """
    supp_feat = seg.embed_frame(supp_frame_u8)
    supp_body = body_mask2d(supp_frame_u8, body_thresh, body_min_px)
    Pos_n, Neg_n = extract_support_vectors_bodymasked(supp_feat, supp_mask2d, supp_body,
                                                       thr_hi, thr_lo)
    radius_px = radius_k * mask_bbox_diag(supp_mask2d)

    best = None  # (score, frame_idx, pos_xy, neg_xy)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, body_thresh, body_min_px)
        pos_xy, neg_xy = pick_points_dense_bodymasked_local(pos_map, neg_map, q_body,
                                                             frame_u8.shape, radius_px)
        score = float((pos_map - neg_map).max())
        if best is None or score > best[0]:
            best = (score, fidx, pos_xy, neg_xy)
    return best[1], best[2], best[3]


def pick_points_dense_bodymasked_similarity(pos_map: np.ndarray, neg_map: np.ndarray,
                                             query_body2d: np.ndarray, img_hw: tuple,
                                             score_thresh: float = 0.0, dilate_iters: int = 2,
                                             max_dilate_iters: int = 8) -> tuple:
    """Like pick_points_dense_bodymasked_local, but the negative search neighborhood is
    derived from the similarity map itself instead of a fixed pixel radius: the
    'object blob' is the connected component of {score_map > score_thresh}
    (score_map = pos_map - neg_map) containing the positive cell, i.e. the region the
    support/query matching itself considers FG-like. The negative point is then the
    best background match in a ring just outside that blob (dilate_iters grid cells),
    intersected with the body mask -- ties negative placement to the actual matching
    geometry (where the object 'ends' according to the embeddings) rather than an
    anatomy-size proxy. If the ring is empty (e.g. blob touches the body-mask edge),
    dilate_iters is doubled up to max_dilate_iters; if still empty, falls back to the
    whole body mask (same fallback as pick_points_dense_bodymasked). The positive point
    (argmax(pos_map - neg_map), on the ORIGINAL unmasked neg_map) is unchanged."""
    from scipy.ndimage import binary_dilation
    from skimage.measure import label as cc_label

    h, w = pos_map.shape
    H, W = img_hw

    def to_xy(idx):
        py, px = idx
        return (float((px + 0.5) / w * W), float((py + 0.5) / h * H))

    score_map = pos_map - neg_map
    pos_idx = np.unravel_index(np.argmax(score_map), score_map.shape)
    pos_xy = to_xy(pos_idx)

    body = torch.from_numpy(query_body2d.astype(np.float32))[None, None]
    body = F.interpolate(body, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    body_grid = (body.numpy() > 0.5)

    blob_seed = score_map > score_thresh
    labeled = cc_label(blob_seed)
    comp_id = labeled[pos_idx]
    blob = np.zeros_like(blob_seed)
    if comp_id != 0:
        blob = (labeled == comp_id)
    else:
        blob[pos_idx] = True

    n_iter = dilate_iters
    candidates = body_grid
    while n_iter <= max_dilate_iters:
        dilated = binary_dilation(blob, iterations=n_iter)
        ring = dilated & (~blob) & body_grid
        if ring.any():
            candidates = ring
            break
        n_iter *= 2

    neg_map_masked = np.where(candidates, neg_map, -1.0)
    neg_idx = np.unravel_index(np.argmax(neg_map_masked), neg_map_masked.shape)

    return pos_xy, to_xy(neg_idx)


def support_prompt_for_query_dense_bodymasked_similarity(seg, supp_frame_u8: np.ndarray,
                                                          supp_mask2d: np.ndarray, query_frames: list,
                                                          thr_hi: float = 0.7, thr_lo: float = 0.3,
                                                          body_thresh: float = 10.0,
                                                          body_min_px: int = 50,
                                                          score_thresh: float = 0.0,
                                                          dilate_iters: int = 2,
                                                          max_dilate_iters: int = 8) -> tuple:
    """Similarity-neighborhood counterpart of support_prompt_for_query_dense_bodymasked_local:
    same body-masked positive/background bags, but the negative point search per query
    frame is restricted to the ring just outside the score_map's own object blob (see
    pick_points_dense_bodymasked_similarity) instead of a bbox-derived pixel radius. Same
    call signature/return shape, added alongside the other support_prompt_for_query_dense_*
    variants (not a replacement).

    returns: (frame_idx, pos_xy, neg_xy)
    """
    supp_feat = seg.embed_frame(supp_frame_u8)
    supp_body = body_mask2d(supp_frame_u8, body_thresh, body_min_px)
    Pos_n, Neg_n = extract_support_vectors_bodymasked(supp_feat, supp_mask2d, supp_body,
                                                       thr_hi, thr_lo)

    best = None  # (score, frame_idx, pos_xy, neg_xy)
    for fidx, frame_u8 in query_frames:
        feat = seg.embed_frame(frame_u8)
        pos_map, neg_map = dense_similarity_maps(feat, Pos_n, Neg_n)
        q_body = body_mask2d(frame_u8, body_thresh, body_min_px)
        pos_xy, neg_xy = pick_points_dense_bodymasked_similarity(
            pos_map, neg_map, q_body, frame_u8.shape, score_thresh, dilate_iters, max_dilate_iters)
        score = float((pos_map - neg_map).max())
        if best is None or score > best[0]:
            best = (score, fidx, pos_xy, neg_xy)
    return best[1], best[2], best[3]
