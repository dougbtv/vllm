# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HF config for DiffusionGemma4 (not yet upstream in transformers)."""

from typing import Any

from transformers import PretrainedConfig


class DiffusionGemma4TextConfig(PretrainedConfig):
    model_type = "diffusion_gemma4_text"

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        # DiffusionGemma4 always uses MoE and K=V sharing for full_attention
        # layers. The HF reference removed these config fields entirely.
        if getattr(self, "num_experts", None):
            self.enable_moe_block = True
        self.attention_k_eq_v = True


class DiffusionGemma4Config(PretrainedConfig):
    model_type = "diffusion_gemma4"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        canvas_length: int = 256,
        self_conditioning_size: int | None = None,
        **kwargs: Any,
    ):
        text_config = text_config or {}
        self.text_config = DiffusionGemma4TextConfig(**text_config)
        self.canvas_length = canvas_length
        self.self_conditioning_size = self_conditioning_size
        super().__init__(**kwargs)
