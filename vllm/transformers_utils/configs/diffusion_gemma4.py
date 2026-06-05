# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for DiffusionGemma (not yet upstream in transformers).

RC0.1 is the final form and uses the ``diffusion_gemma`` model_type (no "4").
The ``diffusion_gemma4`` classes below are kept only to load earlier RC0/test
checkpoints (including the NVFP4/FP8 quantized test checkpoints).
"""

from typing import Any

from transformers import PretrainedConfig
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig


def _init_text_config(self: PretrainedConfig, **kwargs: Any) -> None:
    PretrainedConfig.__init__(self, **kwargs)
    # DiffusionGemma always uses MoE and K=V sharing for full_attention
    # layers. The HF reference removed these config fields entirely.
    if getattr(self, "num_experts", None):
        self.enable_moe_block = True
    self.attention_k_eq_v = True


def _init_config(
    self: PretrainedConfig,
    text_config_class: type[PretrainedConfig],
    text_config: dict[str, Any] | None,
    canvas_length: int,
    self_conditioning_size: int | None,
    **kwargs: Any,
) -> None:
    self.text_config = text_config_class(**(text_config or {}))
    self.canvas_length = canvas_length
    self.self_conditioning_size = self_conditioning_size
    # Convert vision_config dict to a proper config object so that
    # downstream code (AutoModel.from_config, ProcessingInfo, etc.)
    # can access attributes normally.
    vision_config = kwargs.pop("vision_config", None)
    if isinstance(vision_config, dict):
        # Use Gemma4VisionConfig (not bare PretrainedConfig) so that
        # AutoModel.from_config() can resolve the vision tower model class.
        # DiffusionGemma's vision tower is architecturally identical to Gemma4's.
        self.vision_config = Gemma4VisionConfig(**vision_config)
    else:
        self.vision_config = vision_config
    PretrainedConfig.__init__(self, **kwargs)


class DiffusionGemmaTextConfig(PretrainedConfig):
    model_type = "diffusion_gemma_text"

    def __init__(self, **kwargs: Any):
        _init_text_config(self, **kwargs)


class DiffusionGemmaConfig(PretrainedConfig):
    model_type = "diffusion_gemma"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        _init_config(
            self,
            DiffusionGemmaTextConfig,
            text_config,
            canvas_length,
            self_conditioning_size,
            **kwargs,
        )


# TODO(diffusion): backward-compat for the pre-RC0.1 "diffusion_gemma4" naming
# (used by RC0 / v2 / the NVFP4+FP8 test checkpoints). Remove these "...4"
# classes once all checkpoints are finalized to the RC0.1 "diffusion_gemma"
# naming.
class DiffusionGemma4TextConfig(PretrainedConfig):
    model_type = "diffusion_gemma4_text"

    def __init__(self, **kwargs: Any):
        _init_text_config(self, **kwargs)


class DiffusionGemma4Config(PretrainedConfig):
    model_type = "diffusion_gemma4"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        _init_config(
            self,
            DiffusionGemma4TextConfig,
            text_config,
            canvas_length,
            self_conditioning_size,
            **kwargs,
        )
