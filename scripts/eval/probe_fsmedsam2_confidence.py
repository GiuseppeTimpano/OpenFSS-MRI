"""
Probe: does SAM2's native `object_score_logits` (computed by every video-predictor
frame but discarded by FS_MedSAM2's own inference scripts) correlate with per-slice
Dice during FS_MedSAM2-style mask-prompt propagation?

Isolated experiment, no edits to third_party/MedSAM2 or third_party/FS_MedSAM2.
FS_MedSAM2's own `sam2/` folder is NOT put on sys.path as a package (it collides
with the real installed `sam2` package under third_party/MedSAM2 -- both are named
`sam2`). Instead its two files are loaded by path and registered into the REAL
`sam2` package's module namespace, so their internal `from sam2.modeling...` /
`from sam2.utils.misc...` imports resolve against third_party/MedSAM2 (which is
already fully installed, checkpoints/configs included) rather than needing
FS_MedSAM2 merged into a sam2 checkout as its README assumes.

Usage:
  python -m scripts.eval.probe_fsmedsam2_confidence \
      --data_dir <processed_dir> --label_val 1 --label_name SA \
      --medsam2_ckpt <tiny.pt> --sam2_cfg sam2.1_hiera_t512 \
      --out_csv results/fsmedsam2_probe/SA.csv
"""
import argparse
import csv
import importlib.util
import os
import random
import sys
import types

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_MEDSAM2_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'MedSAM2')
_FSMEDSAM2_ROOT = os.path.join(_REPO_ROOT, 'third_party', 'FS_MedSAM2', 'sam2')


def _load_module_from_path(dotted_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(dotted_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _patched_run_memory_encoder(self, inference_state, frame_idx, batch_size,
                                 high_res_masks, is_mask_from_pts):
    """Version-drift shim: FS_MedSAM2's `_run_memory_encoder` was written against
    an older SAM2 API, before `_encode_new_memory` gained the required
    `object_score_logits` argument (added upstream alongside `no_obj_embed_spatial`).
    The real vendored SAM2Base (third_party/MedSAM2) requires it. FS_MedSAM2 never
    modeled "object absent" frames, so we pass an always-positive score here --
    `is_obj_appearing = (object_score_logits > 0)` stays True, i.e. zero suppression,
    reproducing FS_MedSAM2's original (pre-object_score_logits) behavior exactly.
    Isolated monkeypatch applied only to our predictor instance -- no edit to the
    FS_MedSAM2 submodule itself.
    """
    _, _, current_vision_feats, _, feat_sizes = self._get_image_feature(
        inference_state, frame_idx, batch_size
    )
    object_score_logits = torch.full(
        (batch_size, 1), 10.0, device=high_res_masks.device, dtype=torch.float32
    )
    maskmem_features, maskmem_pos_enc = self._encode_new_memory(
        current_vision_feats=current_vision_feats,
        feat_sizes=feat_sizes,
        pred_masks_high_res=high_res_masks,
        object_score_logits=object_score_logits,
        is_mask_from_pts=is_mask_from_pts,
    )
    storage_device = inference_state["storage_device"]
    maskmem_features = maskmem_features.to(torch.bfloat16)
    maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
    maskmem_pos_enc = self._get_maskmem_pos_enc(
        inference_state, {"maskmem_pos_enc": maskmem_pos_enc}
    )
    return maskmem_features, maskmem_pos_enc


def _patched_run_single_frame_inference(self, inference_state, output_dict, frame_idx,
                                         batch_size, is_init_cond_frame, point_inputs,
                                         mask_inputs, reverse, run_mem_encoder,
                                         prev_sam_mask_logits=None):
    """Version-drift shim: `track_step` (inherited from the real vendored SAM2Base)
    already computes and stores `object_score_logits` in its `current_out`, but
    FS_MedSAM2's own `_run_single_frame_inference` drops it when building
    `compact_current_out` (never plumbed through in their older-API fork). This is
    exactly the "free, already-computed, discarded" confidence signal this probe
    is measuring -- we just stop throwing it away. Same body as the original
    method, plus one extra key. Isolated monkeypatch, no edit to the submodule."""
    from sam2.utils.misc import fill_holes_in_mask_scores as _fill_holes

    _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = \
        self._get_image_feature(inference_state, frame_idx, batch_size)

    assert point_inputs is None or mask_inputs is None
    current_out = self.track_step(
        frame_idx=frame_idx,
        is_init_cond_frame=is_init_cond_frame,
        current_vision_feats=current_vision_feats,
        current_vision_pos_embeds=current_vision_pos_embeds,
        feat_sizes=feat_sizes,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        output_dict=output_dict,
        num_frames=inference_state["num_frames"],
        track_in_reverse=reverse,
        run_mem_encoder=run_mem_encoder,
        prev_sam_mask_logits=prev_sam_mask_logits,
    )

    storage_device = inference_state["storage_device"]
    maskmem_features = current_out["maskmem_features"]
    if maskmem_features is not None:
        maskmem_features = maskmem_features.to(torch.bfloat16)
        maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
    pred_masks_gpu = current_out["pred_masks"]
    if self.fill_hole_area > 0:
        pred_masks_gpu = _fill_holes(pred_masks_gpu, self.fill_hole_area)
    pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
    maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
    obj_ptr = current_out["obj_ptr"]
    object_score_logits = current_out.get("object_score_logits")
    if object_score_logits is not None:
        object_score_logits = object_score_logits.to(storage_device, non_blocking=True)
    compact_current_out = {
        "maskmem_features": maskmem_features,
        "maskmem_pos_enc": maskmem_pos_enc,
        "pred_masks": pred_masks,
        "obj_ptr": obj_ptr,
        "object_score_logits": object_score_logits,
    }
    return compact_current_out, pred_masks_gpu


def _build_fsmedsam2_predictor(model_cfg: str, ckpt_path: str, device: str):
    """Loads FS_MedSAM2's predictor class against the REAL `sam2` package
    (third_party/MedSAM2), without ever putting FS_MedSAM2/sam2 on sys.path."""
    if _MEDSAM2_ROOT not in sys.path:
        sys.path.insert(0, _MEDSAM2_ROOT)
    import sam2  # noqa: F401 -- real package; its __init__ sets up Hydra config search path

    # Register FS_MedSAM2's two extra files as if they lived inside the real
    # `sam2` package, so their `from sam2.X import Y` lines resolve correctly.
    _load_module_from_path(
        'sam2.utils.misc_fsmedsam2',
        os.path.join(_FSMEDSAM2_ROOT, 'utils', 'misc_fsmedsam2.py'))
    _load_module_from_path(
        'sam2.sam2_video_predictor_fsmedsam2',
        os.path.join(_FSMEDSAM2_ROOT, 'sam2_video_predictor_fsmedsam2.py'))
    build_mod = _load_module_from_path(
        'sam2.build_fsmedsam2',
        os.path.join(_FSMEDSAM2_ROOT, 'build_fsmedsam2.py'))

    predictor = build_mod.build_fsmedsam2_video_predictor(model_cfg, ckpt_path, device=device)
    predictor._run_memory_encoder = types.MethodType(_patched_run_memory_encoder, predictor)
    predictor._run_single_frame_inference = types.MethodType(
        _patched_run_single_frame_inference, predictor)
    return predictor


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def _key_slice(fg_mask: np.ndarray) -> int:
    """Largest-area FG slice (same convention as models/support_prompt.py:key_slice)."""
    areas = fg_mask.reshape(fg_mask.shape[0], -1).sum(1)
    return int(np.argmax(areas))


def _resize_mask_nearest(mask: np.ndarray, out_hw: tuple) -> np.ndarray:
    """`init_state_by_np_data` records video_height/width from the ALREADY-RESIZED
    (image_size x image_size) input array, not the original slice shape -- so
    propagate_in_video's output masks come back at image_size resolution, not the
    original H,W. Resize back down (nearest, since it's a binary mask) before Dice."""
    h, w = out_hw
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.array(img) > 0


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = float((pred & gt).sum())
    denom = float(pred.sum() + gt.sum())
    return 1.0 if denom == 0 else 2 * inter / denom


def probe(data_dir: str, label_val: int, label_name: str, medsam2_ckpt: str,
          sam2_cfg: str, device: str, out_csv: str, seed: int, max_scans: int):
    from models.medsam2_adapter import resize_grayscale_to_rgb_and_resize, volume_to_uint8

    predictor = _build_fsmedsam2_predictor(sam2_cfg, medsam2_ckpt, device)

    paths = sorted(p for p in os.listdir(data_dir) if p.startswith('image_'))
    sids = [p.replace('image_', '').replace('.nii.gz', '') for p in paths]

    fg_by_sid = {}
    for sid in sids:
        lbl = _read_nii(os.path.join(data_dir, f'label_{sid}.nii.gz'))
        fg = (lbl == label_val).astype(np.uint8)
        if fg.any():
            fg_by_sid[sid] = fg
    if len(fg_by_sid) < 2:
        raise ValueError(f'need >=2 scans with label {label_val} in {data_dir}')

    rng = random.Random(seed)
    query_sids = list(fg_by_sid)
    if max_scans:
        query_sids = query_sids[:max_scans]

    rows = []
    for qsid in query_sids:
        pool = [s for s in fg_by_sid if s != qsid]
        supp_sid = rng.choice(pool)

        q_img = _read_nii(os.path.join(data_dir, f'image_{qsid}.nii.gz')).astype(np.float32)
        q_fg = fg_by_sid[qsid]
        fg_idx = np.where(q_fg.any(axis=(1, 2)))[0]
        z0, z1 = int(fg_idx.min()), int(fg_idx.max())
        q_u8 = volume_to_uint8(q_img)[z0:z1 + 1]
        q_fg_crop = q_fg[z0:z1 + 1]

        supp_img = _read_nii(os.path.join(data_dir, f'image_{supp_sid}.nii.gz')).astype(np.float32)
        supp_fg = fg_by_sid[supp_sid]
        supp_u8 = volume_to_uint8(supp_img)
        supp_z = _key_slice(supp_fg)
        supp_mask_2d = supp_fg[supp_z].astype(bool)
        supp_slice_u8 = supp_u8[supp_z:supp_z + 1]

        n_query = q_u8.shape[0]
        left_idx = n_query // 2
        right_idx = left_idx + 1

        all_u8 = np.concatenate([q_u8[:left_idx], supp_slice_u8, q_u8[left_idx:]], axis=0)
        images_np = resize_grayscale_to_rgb_and_resize(all_u8, predictor.image_size) / 255.0

        inference_state = predictor.init_state_by_np_data(images_np=images_np)
        predictor.reset_state(inference_state)
        predictor.add_new_mask(inference_state=inference_state, frame_idx=left_idx,
                                obj_id=1, mask=supp_mask_2d)

        video_segments = {}
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=right_idx):
            video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
                inference_state, start_frame_idx=left_idx - 1, reverse=True):
            video_segments[out_frame_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()

        output_dict = inference_state['output_dict']

        for j in range(all_u8.shape[0]):
            if j == left_idx:
                continue  # the support frame itself, not a query prediction
            q_j = j if j < left_idx else j - 1  # index back into q_fg_crop / q_u8
            pred = video_segments.get(j)
            if pred is None:
                continue
            gt = q_fg_crop[q_j].astype(bool)
            pred_resized = _resize_mask_nearest(pred.squeeze().astype(bool), gt.shape)
            dice = _dice(pred_resized, gt)

            out = (output_dict['cond_frame_outputs'].get(j) or
                   output_dict['non_cond_frame_outputs'].get(j))
            osl = out.get('object_score_logits') if out is not None else None
            score = float(osl.reshape(-1)[0]) if osl is not None else float('nan')

            rows.append({
                'query_scan': qsid, 'support_scan': supp_sid, 'label': label_name,
                'frame_idx': j, 'dist_from_support': abs(j - left_idx),
                'object_score_logit': score, 'dice': dice,
            })
        print(f'{qsid}: {len(rows)} rows so far (support={supp_sid})')

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['query_scan', 'support_scan', 'label',
                                                'frame_idx', 'dist_from_support',
                                                'object_score_logit', 'dice'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'wrote {len(rows)} rows to {out_csv}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--label_val', type=int, required=True)
    ap.add_argument('--label_name', required=True)
    ap.add_argument('--medsam2_ckpt', required=True)
    ap.add_argument('--sam2_cfg', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--out_csv', required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max_scans', type=int, default=0, help='0 = all scans with this label')
    args = ap.parse_args()
    probe(args.data_dir, args.label_val, args.label_name, args.medsam2_ckpt,
          args.sam2_cfg, args.device, args.out_csv, args.seed, args.max_scans)
