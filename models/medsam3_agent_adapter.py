"""
Bridges this project's LoRA-loaded MedSAM3 (models/medsam3_adapter.py) to the
upstream agent loop (third_party/MedSAM3/sam3/agent/), which ships disconnected
from the LoRA weights: agent_core.agent_inference() expects a pre-bound
call_sam_service + send_generate_request. Done here via functools.partial.

LLM backend (sam3/agent/client_llm.py, OpenAI-compatible endpoint):
  - real OpenAI ("gpt-4o"): OPENAI_API_KEY env, llm_server_url=None.
  - self-hosted (e.g. vLLM): llm_server_url="http://host:port/v1".
Real external cost per image (up to max_generations round-trips) -- smoke
test single slices first (scripts/agent/demo_medsam3_agent.py).

File-based upstream: agent loop reads image *paths*, not in-memory arrays --
write each slice to disk before calling segment_image().
"""
import functools
import os
import sys

import torch

from models.medsam3_adapter import _REPO_ROOT, build_medsam3_lora_model


class MedSAM3AgentSegmenter:
    """Wraps the same LoRA-patched MedSAM3 model as MedSAM3Segmenter, but
    drives it through the upstream multi-round agent loop (LLM decides when
    to re-prompt / accept / reject masks) instead of one forward pass.
    Single 2D image API only -- see module docstring."""

    def __init__(
        self,
        config_path: str | None = None,
        weights_path: str | None = None,
        resolution: int = 1008,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        llm_model: str = "gpt-4o",
        llm_server_url: str | None = None,
        llm_api_key: str | None = None,
        max_generations: int = 20,
    ):
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)

        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.agent.agent_core import agent_inference
        from sam3.agent.client_sam3 import call_sam_service
        from sam3.agent.client_llm import send_generate_request

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        model = build_medsam3_lora_model(config_path, weights_path, self.device)

        self.processor = Sam3Processor(
            model, resolution=resolution, device=self.device.type,
            confidence_threshold=confidence_threshold,
        )
        self._agent_inference = agent_inference
        self._call_sam_service = functools.partial(call_sam_service, self.processor)
        self._send_generate_request = functools.partial(
            send_generate_request, server_url=llm_server_url, model=llm_model,
            api_key=llm_api_key,
        )
        self.max_generations = max_generations

    def segment_image(self, image_path: str, text_prompt: str, output_dir: str,
                       debug: bool = False):
        """
        image_path  : path to a 2D image ALREADY ON DISK (PNG/JPG).
        text_prompt : initial natural-language query, e.g. "segment the liver".
        output_dir  : where the agent writes intermediate SAM3 outputs, debug
                      history, and per-round visualizations.
        returns     : (agent_history, final_outputs_dict, rendered_PIL_image) --
                      final_outputs_dict has pred_boxes/pred_masks (RLE)/
                      pred_scores, see sam3/agent/agent_core.py:agent_inference.
        """
        return self._agent_inference(
            image_path, text_prompt, debug=debug,
            send_generate_request=self._send_generate_request,
            call_sam_service=self._call_sam_service,
            max_generations=self.max_generations,
            output_dir=output_dir,
        )
