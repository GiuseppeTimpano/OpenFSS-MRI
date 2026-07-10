import glob
import json
import os
import re

import numpy as np
import SimpleITK as sitk


def n4_bias_field_correction(image: sitk.Image, shrink_factor=4, n_iters=[50, 40, 30, 20]) -> sitk.Image:
    img_float = sitk.Cast(image, sitk.sitkFloat32)
    img_shrunk = sitk.Shrink(img_float, [shrink_factor] * img_float.GetDimension())

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(n_iters)
    corrector.Execute(img_shrunk)

    log_bias_field = corrector.GetLogBiasFieldAsImage(img_float)
    corrected_full = img_float / sitk.Exp(log_bias_field)

    return corrected_full


def intensity_clip_upper(image: sitk.Image, upper_percentile: float = 99.5) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image)
    high = np.percentile(arr, upper_percentile)
    arr[arr > high] = high
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    return out


def resample_help_function(image_itk: sitk.Image,
                           new_spacing: list,
                           is_label: bool,
                           interpolator=None) -> sitk.Image:
    original_spacing = image_itk.GetSpacing()
    original_size = image_itk.GetSize()

    out_size = [
        int(original_size[0] * (original_spacing[0] / new_spacing[0]) + 1),
        int(original_size[1] * (original_spacing[1] / new_spacing[1]) + 1),
        int(original_size[2] * (original_spacing[2] / new_spacing[2]) + 1),
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(image_itk.GetDirection())
    resample.SetOutputOrigin(image_itk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)

    if interpolator is not None:
        resample.SetInterpolator(interpolator)
    elif is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkBSpline)

    return resample.Execute(image_itk)


def resample_label_onehot(label_itk: sitk.Image, new_spacing: list) -> sitk.Image:
    arr = sitk.GetArrayFromImage(label_itk)
    label_vals = np.unique(arr)
    out_vol = None

    for lbv in label_vals:
        mask = sitk.GetImageFromArray(np.float32(arr == lbv))
        mask.CopyInformation(label_itk)
        resampled = resample_help_function(mask, new_spacing, is_label=False,
                                           interpolator=sitk.sitkLinear)
        resampled_arr = np.rint(sitk.GetArrayFromImage(resampled)).astype(np.int32) * int(lbv)
        if out_vol is None:
            out_vol = resampled_arr
        else:
            out_vol[resampled_arr == int(lbv)] = int(lbv)

    result = sitk.GetImageFromArray(out_vol)
    result.SetSpacing(new_spacing)
    result.SetOrigin(label_itk.GetOrigin())
    result.SetDirection(label_itk.GetDirection())
    return result


def center_crop_2d(image_itk: sitk.Image, crop_size: int = 256, padval: float = 0.0) -> sitk.Image:
    arr = sitk.GetArrayFromImage(image_itk)  # [Z, Y, X]
    cy, cx = arr.shape[1] // 2, arr.shape[2] // 2
    half = (crop_size + 1) // 2

    out = np.full((arr.shape[0], crop_size + 1, crop_size + 1), fill_value=padval, dtype=arr.dtype)

    y_start = max(0, cy - half)
    y_end   = min(arr.shape[1], cy + half)
    x_start = max(0, cx - half)
    x_end   = min(arr.shape[2], cx + half)

    dst_y0 = half - (cy - y_start)
    dst_y1 = dst_y0 + (y_end - y_start)
    dst_x0 = half - (cx - x_start)
    dst_x1 = dst_x0 + (x_end - x_start)

    out[:, dst_y0:dst_y1, dst_x0:dst_x1] = arr[:, y_start:y_end, x_start:x_end]
    out = out[:, :crop_size, :crop_size]

    result = sitk.GetImageFromArray(out)
    result.SetSpacing(image_itk.GetSpacing())
    result.SetOrigin(image_itk.GetOrigin())
    result.SetDirection(image_itk.GetDirection())
    return result


def crop_to_label_bbox_2d(image_itk: sitk.Image, label_itk: sitk.Image,
                          bbox_source_itk: sitk.Image = None,
                          margin_px: int = 40,
                          mask_background: bool = True) -> tuple:
    """Crop image+label to the in-plane bbox of bbox_source_itk's nonzero voxels (union
    over all Z, default: label_itk itself), + margin_px on each side. Same crop window
    applied to every slice.

    For single-leg-per-volume datasets whose raw FOV still shows both legs while only
    one is annotated: bbox_source unambiguously marks which leg is real, so this
    removes the other, unannotated leg once offline -- no runtime heuristic (leg size,
    position) can do this reliably, since the annotated leg is not always the bigger
    one in frame. Pass the dataset's whole-muscle+SAT mask as bbox_source when
    available: it's a filled silhouette of the whole annotated leg, so its bbox is
    tighter/more robust than one derived from a handful of discrete muscle blobs
    (label_itk), which can leave gaps near the leg's true boundary.

    A rectangular bbox alone does not guarantee the other, unannotated leg is fully
    excluded -- it can still fall inside the margin, or overlap the crop window when
    both legs sit close together. If mask_background (default True) and bbox_source_itk
    is given, every image voxel outside bbox_source's own per-voxel mask (not just its
    2D bbox) is zeroed after cropping -- i.e. only pixels that are muscle or SAT/fat
    survive, per-slice, not only within the bbox rectangle."""
    img_arr = sitk.GetArrayFromImage(image_itk)  # [Z, Y, X]
    lbl_arr = sitk.GetArrayFromImage(label_itk)
    src_arr = sitk.GetArrayFromImage(bbox_source_itk) if bbox_source_itk is not None else lbl_arr

    mask2d = (src_arr > 0).any(axis=0)  # [Y, X]
    ys, xs = np.where(mask2d)
    y0 = max(0, int(ys.min()) - margin_px)
    y1 = min(img_arr.shape[1], int(ys.max()) + 1 + margin_px)
    x0 = max(0, int(xs.min()) - margin_px)
    x1 = min(img_arr.shape[2], int(xs.max()) + 1 + margin_px)

    def _wrap(arr, ref):
        out = sitk.GetImageFromArray(arr)
        out.SetSpacing(ref.GetSpacing())
        out.SetOrigin(ref.GetOrigin())
        out.SetDirection(ref.GetDirection())
        return out

    cropped_img_arr = img_arr[:, y0:y1, x0:x1]
    cropped_lbl_arr = lbl_arr[:, y0:y1, x0:x1]

    if mask_background and bbox_source_itk is not None:
        cropped_src_arr = src_arr[:, y0:y1, x0:x1]
        cropped_img_arr = cropped_img_arr.copy()
        cropped_img_arr[cropped_src_arr == 0] = 0

    cropped_img = _wrap(cropped_img_arr, image_itk)
    cropped_lbl = _wrap(cropped_lbl_arr, label_itk)
    return cropped_img, cropped_lbl


def build_gt_classmap(label_dir: str,
                      label_names: list,
                      min_pixels_list: list,
                      out_dir: str) -> None:
    label_paths = sorted(
        glob.glob(os.path.join(label_dir, 'label_*.nii.gz')),
        key=lambda x: int(re.findall(r'\d+', os.path.basename(x))[-1]),
    )
    if not label_paths:
        raise FileNotFoundError(f'No label_*.nii.gz in {label_dir}')

    for min_px in min_pixels_list:
        classmap = {name: {} for name in label_names}

        for lbl_path in label_paths:
            scan_id = re.findall(r'\d+', os.path.basename(lbl_path))[-1]
            lbl_vol = sitk.GetArrayFromImage(sitk.ReadImage(lbl_path))

            for name in label_names:
                classmap[name][scan_id] = []

            for z in range(lbl_vol.shape[0]):
                slc = lbl_vol[z]
                slice_sum = int(np.sum(slc))
                for cls_idx, name in enumerate(label_names):
                    if cls_idx in slc and slice_sum >= min_px:
                        classmap[name][scan_id].append(z)

        out_path = os.path.join(out_dir, f'gt_classmap_{min_px}.json')
        with open(out_path, 'w') as f:
            json.dump(classmap, f)
        print(f'gt_classmap_{min_px}.json written')

def z_volume_norm(image_itk: sitk.Image) -> sitk.Image:
    """Per-volume z-score normalization. Returns float32 sitk.Image."""
    arr = sitk.GetArrayFromImage(image_itk).astype(np.float32)
    m, s = arr.mean(), arr.std()
    arr = (arr - m) / (s + 1e-8)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image_itk)
    return out
