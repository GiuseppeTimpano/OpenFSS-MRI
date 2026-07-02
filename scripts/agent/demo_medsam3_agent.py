"""
Single-slice smoke test for the MedSAM3 agent loop (models/medsam3_agent_adapter.py).
NOT a full eval -- one scan, one slice, one organ; build eval_medsam3_agent.py
(mirroring eval_medsam3.py) once this works and LLM cost/latency is known.

Example:
  PYTHONPATH=. .venv/bin/python scripts/agent/demo_medsam3_agent.py \\
      --target_data_dir data/datasets/CIRRMR/processed/T1 --test_label 1 \\
      --llm_model gpt-4o   # needs OPENAI_API_KEY
"""
import argparse
import glob
import os

import numpy as np
import SimpleITK as sitk
from PIL import Image as PILImage

from models.medsam3_adapter import PROMPT_TEMPLATES, volume_to_uint8
from models.medsam3_agent_adapter import MedSAM3AgentSegmenter


def _read_nii(path: str) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data_dir', type=str, required=True,
                         help='processed dir with image_*/label_*.nii.gz')
    parser.add_argument('--test_label', type=int, required=True,
                         help='label id, e.g. 1 = LIVER (see configs/*.yaml label_names)')
    parser.add_argument('--label_name', type=str, default='LIVER',
                         help='key into models.medsam3_adapter.PROMPT_TEMPLATES')
    parser.add_argument('--scan_index', type=int, default=0,
                         help='which scan in the dir to use (sorted order)')
    parser.add_argument('--medsam3_config', type=str, default=None)
    parser.add_argument('--medsam3_weights', type=str, default=None)
    parser.add_argument('--llm_model', type=str, default='gpt-4o')
    parser.add_argument('--llm_server_url', type=str, default=None,
                         help='OpenAI-compatible endpoint, e.g. http://host:port/v1; '
                              'omit to use the real OpenAI API (needs OPENAI_API_KEY)')
    parser.add_argument('--llm_api_key', type=str, default=None,
                         help='defaults to OPENAI_API_KEY env var if unset')
    parser.add_argument('--max_generations', type=int, default=20)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results/medsam3_agent_demo')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.target_data_dir, 'image_*.nii.gz')))
    if not paths:
        raise ValueError(f'No scans found in {args.target_data_dir}')
    img_path = paths[args.scan_index]
    sid = os.path.basename(img_path).replace('image_', '').replace('.nii.gz', '')
    lbl_path = os.path.join(args.target_data_dir, f'label_{sid}.nii.gz')

    img = _read_nii(img_path).astype(np.float32)
    lbl = _read_nii(lbl_path).astype(np.int32)
    fg = (lbl == args.test_label)
    fg_idx = np.where(fg.any(axis=(1, 2)))[0]
    if len(fg_idx) == 0:
        raise ValueError(f'scan {sid} has no foreground for label {args.test_label}')
    z = int(fg_idx[len(fg_idx) // 2])  # middle FG slice, most representative

    vol_u8 = volume_to_uint8(img)
    slice_u8 = vol_u8[z]

    os.makedirs(args.output_dir, exist_ok=True)
    slice_path = os.path.join(args.output_dir, f'{sid}_z{z}_input.png')
    PILImage.fromarray(slice_u8).convert('RGB').save(slice_path)
    print(f'scan={sid} slice={z} label={args.test_label} -> {slice_path}')

    prompt = PROMPT_TEMPLATES[args.label_name]
    print(f'text prompt: "segment the {prompt}"')

    segmenter = MedSAM3AgentSegmenter(
        config_path=args.medsam3_config,
        weights_path=args.medsam3_weights,
        device=args.device,
        llm_model=args.llm_model,
        llm_server_url=args.llm_server_url,
        llm_api_key=args.llm_api_key,
        max_generations=args.max_generations,
    )

    history, final_outputs, rendered = segmenter.segment_image(
        slice_path, f'segment the {prompt}', args.output_dir, debug=args.debug)

    rendered_path = os.path.join(args.output_dir, f'{sid}_z{z}_agent_result.png')
    rendered.save(rendered_path)
    print(f'\nagent rounds: {len(history)}')
    print(f'final masks kept: {len(final_outputs["pred_masks"])}')
    print(f'rendered result saved to: {rendered_path}')


if __name__ == '__main__':
    main()
