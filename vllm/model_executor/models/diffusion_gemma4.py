# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DiffusionGemma4 model, ModelState, and Sampler for vLLM.

Single Gemma4 backbone run in two modes (like YOCO):
- encoder mode: causal attention, writes KV cache
- decoder mode: bidirectional attention, reads encoder KV, doesn't write

Same weights, same layers. The only decoder-unique component is a
self-conditioning MLP.

Multimodal support: the model always includes a vision tower (shared with
Gemma4). Images are encoded through the vision tower and projected into
the LM embedding space via Gemma4MultimodalEmbedder.

Design doc: docs/design/diffusion_gemma4_summary.md
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
)

# Reuse Gemma4 components directly
from vllm.model_executor.models.gemma4 import (
    Gemma4Model,
)
from vllm.model_executor.models.gemma4_mm import (
    Gemma4DummyInputsBuilder,
    Gemma4ForConditionalGeneration,
    Gemma4MultimodalEmbedder,
    Gemma4MultiModalProcessor,
    Gemma4ProcessingInfo,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.model_states.interface import ModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)
from vllm.model_executor.models.module_mapping import MultiModelKeys
logger = init_logger(__name__)


class DiffusionGemma4SelfConditioning(nn.Module):
    """Gated MLP that processes soft embeddings from the previous denoising step.

    Structurally identical to Gemma4MLP but with self_conditioning_size
    and post_norm without learned scale.
    """

    def __init__(
        self, hidden_size: int, self_conditioning_size: int, eps: float = 1e-6
    ):
        super().__init__()
        self.pre_norm = RMSNorm(hidden_size, eps=eps)
        self.post_norm = RMSNorm(hidden_size, eps=eps, has_weight=False)
        self.gate_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, self_conditioning_size, bias=False)
        self.down_proj = nn.Linear(self_conditioning_size, hidden_size, bias=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        soft_embeds: torch.Tensor,
    ) -> torch.Tensor:
        x = self.pre_norm(soft_embeds)
        sc_signal = self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )
        return self.post_norm(inputs_embeds + sc_signal)



# ---------------------------------------------------------------------------
# Multimodal processing info (overrides Gemma4 config type check)
# ---------------------------------------------------------------------------


class DiffusionGemma4ProcessingInfo(Gemma4ProcessingInfo):
    """Processing info for DiffusionGemma4.

    Overrides ``get_hf_config`` to accept ``DiffusionGemmaConfig``
    (which inherits from ``PretrainedConfig``, not ``Gemma4Config``).
    Supports image and video modalities.
    """

    def get_hf_config(self):
        # DiffusionGemmaConfig doesn't inherit from Gemma4Config, so we
        # accept any PretrainedConfig here.
        return self.ctx.get_hf_config()

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        # DiffusionGemma4 supports image and video inputs.
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self, seq_len: int, mm_counts: Mapping[str, int]
    ) -> Mapping[str, int] | None:
        config = self.get_hf_config()
        vision_config = getattr(config, "vision_config", None)
        if vision_config is None:
            # TODO(diffusion): backward-compat for the pre-RC0.1
            # architecture name. Remove once old checkpoints are gone.
            return {"image": 0, "video": 0}

        return super().get_mm_max_tokens_per_item(seq_len, mm_counts)


@MULTIMODAL_REGISTRY.register_processor(
    Gemma4MultiModalProcessor,
    info=DiffusionGemma4ProcessingInfo,
    dummy_inputs=Gemma4DummyInputsBuilder,
)
class DiffusionGemma4ForConditionalGeneration(
    nn.Module,
    SupportsMultiModal,
    SupportsPP,
):
    """DiffusionGemma4 for vLLM.

    Single Gemma4 backbone that switches between encoder and decoder mode.
    The encoder path uses standard Gemma4 layers (causal attention, KV write).
    The decoder path uses the same weights with bidirectional attention and
    KV read-only, plus self-conditioning.

    Always includes a vision tower (same as Gemma4) for image understanding.

    In practice, the model's forward() dispatches based on the `mode` kwarg
    set by DiffusionGemma4ModelState.prepare_inputs().
    """

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={
            "model.decoder.": "model.",
            "model.encoder.language_model.": "model.",
        },
        orig_to_new_substr={
            ".experts.": ".moe.experts.",
        },
    )

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    @staticmethod
    def get_model_state_cls():
        return DiffusionGemma4ModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        text_config = vllm_config.model_config.hf_text_config
        self.config = config
        self.model_dtype = vllm_config.model_config.dtype

        # DiffusionGemma4 feeds raw (non-prenormed) input to the MoE router,
        # matching the HF decoder which does router(residual) not
        # router(pre_norm(residual)).
        text_config.router_uses_prenormed_input = False

        # DiffusionGemma4's full-attention layers have NO v_proj — V is
        # computed from k_proj's output (`value_states = key_states` before
        # k_norm in `DiffusionGemma4DecoderTextAttention.forward`). This is
        # the "k_eq_v" variant in our Gemma4 backbone. The checkpoint has no
        # v_proj weights for full-attention layers; without this flag they
        # would silently load with random V projections.
        text_config.attention_k_eq_v = True

        # ---- Vision tower ----
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            self.vision_tower = AutoModel.from_config(
                config=vision_config
            )
            self.embed_vision = Gemma4MultimodalEmbedder(
                vision_config,
                text_config,
                prefix=maybe_prefix(prefix, "embed_vision"),
            )
        else:
            # TODO(diffusion): backward-compat for the pre-RC0.1
            # architecture name. Remove once old checkpoints are gone.
            self.vision_tower = None
            self.embed_vision = None

        # ---- Language backbone (Gemma4Model) ----
        # Use maybe_prefix to ensure correct weight name prefixes for
        # quantization. The quantization config uses hf_to_vllm_mapper to
        # match checkpoint weight names to model parameter names.
        self.model = Gemma4Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        self.lm_head = ParallelLMHead(
            num_embeddings=text_config.vocab_size,
            embedding_dim=text_config.hidden_size,
        )

        if text_config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        # HF DiffusionGemma4 applies the final-logit softcap in fp32, before
        # any other processing. Do it manually in `compute_logits` so the
        # LogitsProcessor only handles the lm_head GEMM.
        self.final_logit_softcapping = getattr(
            text_config, "final_logit_softcapping", None
        )
        self.logits_processor = LogitsProcessor(
            text_config.vocab_size,
            soft_cap=None,
        )

        sc_size = (
            getattr(config, "self_conditioning_size", None)
            or text_config.intermediate_size
        )
        self.self_conditioning = DiffusionGemma4SelfConditioning(
            hidden_size=text_config.hidden_size,
            self_conditioning_size=sc_size,
            eps=getattr(text_config, "rms_norm_eps", 1e-6),
        )

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )



    def compute_self_conditioning(
        self,
        inputs_embeds: torch.Tensor,
        probs: torch.Tensor,
    ) -> torch.Tensor:
        embed_weight = self.model.embed_tokens.weight
        soft_embeds = torch.matmul(
            probs.to(embed_weight.dtype), embed_weight
        ) * self.model.normalizer.to(inputs_embeds.dtype)
        return self.self_conditioning(inputs_embeds, soft_embeds)

    # ------------------------------------------------------------------ #
    # Multimodal: reuse Gemma4's image parsing, processing & embedding
    # ------------------------------------------------------------------ #
    # The vision tower, pooler, embed_vision, and their processing logic
    # are architecturally identical to Gemma4.  Delegate to avoid
    # maintaining a duplicate copy.

    _parse_and_validate_image_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_image_input
    )
    _parse_and_validate_video_input = (
        Gemma4ForConditionalGeneration._parse_and_validate_video_input
    )
    _parse_and_validate_multimodal_inputs = (
        Gemma4ForConditionalGeneration._parse_and_validate_multimodal_inputs
    )
    _encoder_chunk = staticmethod(
        Gemma4ForConditionalGeneration._encoder_chunk
    )
    _process_image_input = (
        Gemma4ForConditionalGeneration._process_image_input
    )
    _process_video_input = (
        Gemma4ForConditionalGeneration._process_video_input
    )
    embed_multimodal = (
        Gemma4ForConditionalGeneration.embed_multimodal
    )

    def get_mm_mapping(self) -> MultiModelKeys:
        """Get the module prefix mapping for multimodal models."""
        return MultiModelKeys.from_string_field(
            language_model="model",
            connector=["embed_vision"],
            tower_model=["vision_tower"],
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            inputs_embeds = None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if logits is not None and self.final_logit_softcapping is not None:
            # HF DiffusionGemma4 casts to fp32 before softcap for numerical
            # stability of the tanh.
            logits = logits.float()
            cap = self.final_logit_softcapping
            logits = torch.tanh(logits / cap) * cap
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        """Load weights from checkpoint.

        Checkpoint layout (HF DiffusionGemma):
          model.encoder.vision_tower.*            → vision tower
          model.encoder.embed_vision.*            → vision embedder
          model.encoder.language_model.layers.*   → backbone
          model.decoder.layers.*                  → backbone (tied)
          model.decoder.embed_tokens.*            → embeddings
          model.decoder.self_conditioning.*       → self-conditioning MLP
          lm_head.*                               → LM head (tied)

        We load encoder weights into our single ``Gemma4Model`` backbone,
        skip duplicate decoder backbone weights, handle vision tower and
        self-conditioning separately.
        """

        sc_params = dict(
            (n, p)
            for n, p in self.named_parameters()
            if n.startswith("self_conditioning.")
        )

        # Collect vision tower + embedder parameters for manual loading.
        vision_params: dict[str, torch.Tensor] = {}
        for n, p in self.named_parameters():
            if n.startswith(("vision_tower.", "embed_vision.")):
                vision_params[n] = p

        def _remap_weights():
            # Use full weight names (including suffixes like .weight_scale, .weight_packed)
            # for deduplication instead of just the base layer name. This is critical for
            # quantized checkpoints where each weight has multiple associated tensors
            # (weight, weight_scale, weight_global_scale, input_global_scale). If we only
            # track base layer names, all the scales would be incorrectly skipped as
            # duplicates, causing the model to produce garbage output.
            seen_weights: set[str] = set()
            for name, weight in weights:
                # Self-conditioning lives under model.decoder.self_conditioning.*
                # in the checkpoint but at self_conditioning.* in our model.
                if "self_conditioning" in name:
                    sc_name = name.split("self_conditioning.", 1)[1]
                    sc_name = "self_conditioning." + sc_name
                    if sc_name in sc_params:
                        sc_params[sc_name].data.copy_(weight)
                    continue

                # Vision tower: model.encoder.vision_tower.* → vision_tower.*
                # In HF, the vision tower is a sibling of language_model
                # under the encoder module.
                if name.startswith("model.encoder.vision_tower."):
                    vt_name = name[len("model.encoder."):]
                    if vt_name in vision_params:
                        vision_params[vt_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Vision tower weight %s (mapped to %s) "
                            "not found in model", name, vt_name)
                    continue

                # Vision embedder: model.encoder.embed_vision.* → embed_vision.*
                if name.startswith("model.encoder.embed_vision."):
                    ev_name = name[len("model.encoder."):]
                    if ev_name in vision_params:
                        vision_params[ev_name].data.copy_(weight)
                    else:
                        logger.warning(
                            "Embed vision weight %s (mapped to %s) "
                            "not found in model", name, ev_name)
                    continue

                # Skip vestigial embed_vision.embedding weights.
                if "embed_vision.embedding." in name:
                    continue

                # Encoder backbone → model.*
                if name.startswith("model.encoder.language_model."):
                    name = name.replace("model.encoder.language_model.", "model.")
                # Decoder backbone → model.* (skip exact duplicates)
                elif name.startswith("model.decoder."):
                    name = name.replace("model.decoder.", "model.")

                # Skip only if we've seen the exact same weight name (including scales)
                if name in seen_weights:
                    continue
                seen_weights.add(name)
                yield name, weight



        # Delegate to Gemma4ForCausalLM.load_weights for the backbone,
        # which handles stacked params, MoE, k_eq_v, etc.
        # Temporarily set self.config to text_config since Gemma4's
        # load_weights expects it (e.g. tie_word_embeddings, layer_types).
        from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

        saved_config = self.config
        self.config = self.model.config
        try:
            Gemma4ForCausalLM.load_weights(self, _remap_weights())
        finally:
            self.config = saved_config

    @classmethod
    def get_placeholder_str(
        cls, modality: str, i: int
    ) -> str | None:
        if modality == "image":
            return "<image_soft_token>"
        if modality == "video":
            return "<|video|>"
        raise ValueError(f"Unsupported modality: {modality}")


DEFAULT_STABILITY_THRESHOLD = 2
DEFAULT_CONFIDENCE_THRESHOLD = 0.005


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)


@torch.compile(dynamic=True)
def _compiled_sample_step(
    # Logits from the model [num_decode * CL, vocab]
    logits: torch.Tensor,
    # Request mapping
    decode_slots: torch.Tensor,     # [num_decode] int64 → slot indices
    decode_idx: torch.Tensor,       # [num_decode] int64 → position in num_reqs
    all_slots: torch.Tensor,        # [num_reqs] int64 → all slot indices
    # State tensors (modified in-place)
    canvas: torch.Tensor,           # [max_num_reqs, CL]
    argmax_canvas: torch.Tensor,    # [max_num_reqs, CL]
    step_tensor: torch.Tensor,      # [max_num_reqs]
    is_encoder_phase: torch.Tensor, # [max_num_reqs]
    confident_tensor: torch.Tensor, # [max_num_reqs]
    sc_probs: torch.Tensor,         # [max_num_reqs, CL, vocab]
    history: torch.Tensor,          # [max_num_reqs, ST, CL]
    history_len_tensor: torch.Tensor,  # [max_num_reqs]
    # Output tensors (modified in-place)
    sampled: torch.Tensor,          # [num_reqs, CL]
    num_sampled: torch.Tensor,      # [num_reqs]
    draft_tokens: torch.Tensor,     # [max_num_reqs, >=CL]
    # Scalar config
    max_denoising_steps: float,
    t_min: float,
    t_max: float,
    confidence_threshold: float,
    vocab_size: int,
    CL: int,
    ST: int,
    # Sampler mode
    use_entropy_bound: bool,
    entropy_bound: float,
    renoise_ratio_modifier: float,
    ar_mask_noise_proportion: float,
    use_autoregressive_mask: bool,
) -> None:
    """Compiled decode step: temperature → Gumbel sample → probs/confidence →
    accept/renoise → convergence, all as vectorized PyTorch ops."""
    num_decode = decode_slots.shape[0]
    device = decode_slots.device

    # ---- Phase 1: Temperature schedule ----
    steps_f = step_tensor[decode_slots].float()
    remaining = (max_denoising_steps - steps_f).clamp(min=1.0)
    temp = t_min + (t_max - t_min) * (remaining / max_denoising_steps)

    # ---- Phase 2: Temperature scaling + Gumbel-max sampling ----
    logits_3d = logits.reshape(num_decode, CL, -1).float()
    scaled = logits_3d / temp[:, None, None].clamp(min=1e-10)

    # Gumbel-max trick: argmax(logits/T + Gumbel) ~ sample from softmax(logits/T)
    u = torch.rand_like(scaled).clamp(min=1e-20)
    gumbel = -torch.log(-torch.log(u))
    # Zero noise when temp==0 (greedy)
    noisy = scaled + gumbel * (temp[:, None, None] > 0).float()
    new_tokens = noisy.view(-1, noisy.shape[-1]).argmax(dim=-1).view(num_decode, CL)
    argmax_tokens = scaled.view(-1, scaled.shape[-1]).argmax(dim=-1).view(
        num_decode, CL
    )

    # ---- Phase 3: Probs, self-conditioning, confidence ----
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()

    token_entropy = -(probs * log_probs).sum(dim=-1)   # [num_decode, CL]
    mean_entropy = token_entropy.mean(dim=-1)           # [num_decode]
    confident_tensor[decode_slots] = mean_entropy < confidence_threshold

    # ---- Phase 4: Entropy-bound acceptance mask (if enabled) ----
    if use_entropy_bound:
        sorted_ent, sorted_idx = torch.sort(token_entropy, dim=-1)
        cumsum_ent = torch.cumsum(sorted_ent, dim=-1)
        cummax_ent = torch.cummax(sorted_ent, dim=-1).values
        sorted_mask = (cumsum_ent - cummax_ent) <= entropy_bound
        eb_mask = torch.zeros_like(sorted_mask)
        eb_mask.scatter_(1, sorted_idx, sorted_mask)

    # ---- Phase 5: Post-sample ----
    is_commit = is_encoder_phase[decode_slots]          # [num_decode]
    is_denoise = ~is_commit
    cur_step = step_tensor[decode_slots].float()

    # Step update: +1 for denoise, reset to 0 for commit
    new_step_val = torch.where(
        is_denoise,
        (cur_step + 1).to(step_tensor.dtype),
        step_tensor.new_zeros(num_decode),
    )
    step_tensor[decode_slots] = new_step_val

    # Random tokens for renoise / canvas reinit
    random_tokens = torch.randint(
        0, vocab_size, (num_decode, CL), device=device, dtype=canvas.dtype
    )

    # Compute denoise canvas (accept/renoise)
    if use_entropy_bound:
        denoise_canvas = torch.where(eb_mask, new_tokens, random_tokens)
    else:
        remaining_d = (max_denoising_steps - cur_step).clamp(min=1.0)
        accept_prob = (1.0 / remaining_d).unsqueeze(1)     # [num_decode, 1]
        renoise_prob = (
            renoise_ratio_modifier
            * (remaining_d - 1.0).clamp(min=0.0)
            / max_denoising_steps
        ).unsqueeze(1)

        cur_canvas = canvas[decode_slots]
        accept_mask = torch.rand(num_decode, CL, device=device) < accept_prob

        if use_autoregressive_mask:
            ar_thresh = ((cur_step + 1.0) / max_denoising_steps).unsqueeze(1)
            pos_ratio = (
                torch.arange(CL, device=device).float()
                * (1.0 - ar_mask_noise_proportion)
                / max(CL - 1, 1)
            )
            accept_mask = accept_mask | (pos_ratio.unsqueeze(0) <= ar_thresh)

        accepted = torch.where(accept_mask, new_tokens, cur_canvas)
        denoise_canvas = torch.where(
            torch.rand(num_decode, CL, device=device) < renoise_prob,
            random_tokens,
            accepted,
        )

    # Canvas: commit → random reinit, denoise → accept/renoise result
    canvas[decode_slots] = torch.where(
        is_commit.unsqueeze(1), random_tokens, denoise_canvas
    )

    # History: write argmax_tokens for denoise requests at circular position
    hist_len = history_len_tensor[decode_slots]
    write_pos = hist_len % ST
    for i in range(ST):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        history[decode_slots, i] = torch.where(
            write_here, argmax_tokens, history[decode_slots, i]
        )

    # Argmax canvas: update for denoise, preserve for commit
    argmax_canvas[decode_slots] = torch.where(
        is_denoise.unsqueeze(1), argmax_tokens, argmax_canvas[decode_slots]
    )

    # History length: increment for denoise, reset for commit
    new_hist_len = torch.where(
        is_denoise, hist_len + 1, hist_len.new_zeros(num_decode)
    )
    history_len_tensor[decode_slots] = new_hist_len

    # SC probs: store for denoise, zero for commit
    sc_probs[decode_slots] = probs.to(sc_probs.dtype) * is_denoise[
        :, None, None
    ].to(sc_probs.dtype)

    # Sampled output: commit → emit argmax_canvas, denoise → 0 (pre-zeroed)
    sampled[decode_idx] = (
        argmax_canvas[decode_slots].to(sampled.dtype)
        * is_commit.unsqueeze(1).to(sampled.dtype)
    )
    num_sampled[decode_idx] = is_commit.to(num_sampled.dtype) * CL

    # ---- Phase 6: Stability + convergence ----
    ref = history[decode_slots, 0]
    mismatch = torch.zeros(num_decode, device=device, dtype=torch.int32)
    for h in range(1, ST):
        mismatch = mismatch + (ref != history[decode_slots, h]).sum(dim=-1).int()
    stable = mismatch == 0

    step_after = step_tensor[decode_slots]
    converged = (stable & confident_tensor[decode_slots] & (new_hist_len >= ST)) | (
        step_after >= max_denoising_steps
    )
    # Commit done → denoise next (False); denoise converged → commit next (True)
    is_encoder_phase[decode_slots] = torch.where(
        is_commit, is_commit.new_zeros(num_decode), converged
    )

    # Overwrite canvas with argmax for newly converged denoise requests
    newly_converged = (converged & is_denoise).unsqueeze(1)
    canvas[decode_slots] = torch.where(
        newly_converged, argmax_canvas[decode_slots], canvas[decode_slots]
    )

    # ---- Phase 7: Copy canvas → draft_tokens for all slots ----
    draft_tokens[all_slots, :CL] = canvas[all_slots]


class DiffusionGemma4RequestStates:
    """Pre-allocated GPU tensors for DiffusionGemma4 per-request state.

    Follows the indexed-slot pattern used by ``RequestState``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
        hidden_size: int,
        stability_threshold: int,
    ):
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps
        self.stability_threshold = stability_threshold
        self.device = device

        self.is_encoder_phase = torch.zeros(
            max_num_reqs, dtype=torch.bool, device=device
        )
        # Canvas tokens [max_num_reqs, canvas_length]
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Step counter (counts up from 0 to max_denoising_steps)
        self.step = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        # Accepted canvas history for stability check
        self.accepted_canvas_history = torch.zeros(
            max_num_reqs,
            stability_threshold,
            canvas_length,
            dtype=torch.int64,
            device=device,
        )
        self.accepted_canvas_history_len = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        # Latest argmax(processed_logits) per slot — what we COMMIT.
        # HF reference commits `argmax_canvas` (argmax of latest step's logits),
        # NOT `current_canvas` (which is the post-renoise stochastic input for
        # the next denoise step). We keep this separate from `canvas` because
        # canvas gets renoised in-place during denoise, while argmax_canvas is
        # the deterministic best-guess we ultimately emit.
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )

        # Per-slot prompt length (set by add_request).
        self.prompt_len = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )

        # Per-slot confidence flag, set by the sampler each step.
        self.confident = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)

        # Per-slot probs from the previous denoise step, set by the sampler.
        self.self_conditioning_probs = torch.zeros(
            max_num_reqs, canvas_length, vocab_size, dtype=torch.float32, device=device
        )

    def init_canvas(self, slot_indices_np: np.ndarray) -> None:
        """Initialize canvas with random tokens for the given slots."""
        n = slot_indices_np.shape[0]
        self.canvas[slot_indices_np] = torch.randint(
            0,
            self.vocab_size,
            (n, self.canvas_length),
            dtype=torch.int64,
            device=self.device,
        )

    def add_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = True
        self.init_canvas(torch.tensor([slot_idx], device=self.device))
        self.step[slot_idx] = 0
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_probs[slot_idx] = 0

    def remove_request(self, slot_idx: int) -> None:
        self.is_encoder_phase[slot_idx] = False
        self.accepted_canvas_history_len[slot_idx] = 0
        self.self_conditioning_probs[slot_idx] = 0

class DiffusionGemma4ModelState(ModelState):
    """ModelState for DiffusionGemma4.

    Single Gemma4 backbone in two modes:
    - encoder mode (num_draft_tokens == 0): causal attention, writes KV
    - decoder mode (num_draft_tokens > 0): bidirectional attention, reads KV
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: Any,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.model = model
        self.device = device

        self.supports_mm_inputs = encoder_cache is not None
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_model_len = self.model_config.max_model_len
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()
        self.dtype = self.model_config.dtype

        if self.supports_mm_inputs:
            from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
            from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
            assert isinstance(encoder_cache, EncoderCache)
            self.encoder_cache = encoder_cache
            self.encoder_runner = EncoderRunner(
                model=self.model,
                max_num_tokens=self.max_num_tokens,
                hidden_size=self.inputs_embeds_size,
                encoder_cache=encoder_cache,
                dtype=self.dtype,
                device=self.device,
            )

        # Per-step MM data produced by get_mm_embeddings and consumed by
        # prepare_inputs.  Stored as raw (mm_embeds, is_mm_embed) so that
        # prepare_inputs can call embed_input_ids directly into the
        # persistent _inputs_embeds_buf, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds: tuple[
            list[torch.Tensor], torch.Tensor
        ] | None = None

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 32
        max_denoising_steps = (
            diffusion_config.max_denoising_steps if diffusion_config else 48
        )

        text_config = self.model_config.hf_text_config
        # Diffusion sampling params come straight from generation_config.json
        # (RC0.1 flat layout); the checkpoint is the source of truth.
        self.gen_config = self.model_config.try_get_generation_config()
        self.diffusion_states = DiffusionGemma4RequestStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            vocab_size=self.model_config.get_vocab_size(),
            max_denoising_steps=max_denoising_steps,
            device=device,
            hidden_size=text_config.hidden_size,
            stability_threshold=self.gen_config["stability_threshold"],
        )
        self._req_id_to_index: dict[str, int] = {}

        # Persistent buffer for per-request causal flags, updated in-place
        # so FULL CUDA graph replay sees the latest values.
        self._causal_buf = torch.zeros(
            self.max_num_reqs, dtype=torch.bool, device=device
        )

        # Persistent inputs_embeds buffer — required so FULL CUDA graph
        # capture and runtime point at the SAME memory address.
        # `prepare_dummy_inputs` (capture path) and `prepare_inputs` (runtime
        # path) both must hand the captured graph a tensor at this address.
        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens,
            text_config.hidden_size,
            dtype=self.model_config.dtype,
            device=device,
        )

        self.canvas_arange = torch.arange(canvas_length, device=device)
        self.decode_slots = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
        self.decode_idx = UvaBackedTensor(self.max_num_reqs, dtype=torch.int64)

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(
        self,
        sampler: Any,
        diffusion_config: Any,
    ) -> tuple[Any, Any] | None:
        gen = self.gen_config
        # Sampler type is derived from the sampler_config class name (nested
        # even in the flat layout): EntropyBound* -> entropy_bound, else ar_euler.
        sampler_cfg = gen.get("sampler_config") or {}
        if "EntropyBound" in sampler_cfg.get("_cls_name", ""):
            sampler_type = "entropy_bound"
            entropy_bound = sampler_cfg.get("entropy_bound")
            if entropy_bound is None or entropy_bound <= 0:
                raise ValueError(
                    f"entropy_bound must be a positive float (got {entropy_bound})"
                )
        else:
            sampler_type = "ar_euler"
            entropy_bound = None
        return DiffusionSampler(
            sampler=sampler,
            diffusion_config=diffusion_config,
            vocab_size=self.model_config.get_vocab_size(),
            diffusion_states=self.diffusion_states,
            t_min=gen["t_min"],
            t_max=gen["t_max"],
            sampler_type=sampler_type,
            entropy_bound=entropy_bound,
            confidence_threshold=gen["confidence_threshold"],
        ), None

    def apply_staged_writes(self) -> None:
        pass

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        if not new_req_data.req_id.startswith("_warmup_"):
            prompt_len = len(new_req_data.prompt_token_ids)
            self.diffusion_states.prompt_len[req_index] = prompt_len

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            self.diffusion_states.remove_request(idx)

    def get_mm_embeddings(self, scheduled_encoder_inputs, input_batch):
        if not self.supports_mm_inputs:
            return None

        mm_hashes, mm_kwargs = self.encoder_runner.prepare_mm_inputs(
            scheduled_encoder_inputs
        )
        if mm_kwargs:
            encoder_outputs = self.encoder_runner.execute_mm_encoder(
                mm_kwargs
            )
            self.encoder_cache.encoder_outputs.update(
                zip(mm_hashes, encoder_outputs)
            )

        mm_embeds, is_mm_embed = self.encoder_runner.gather_mm_embeddings(
            input_batch.req_ids,
            input_batch.num_tokens,
            input_batch.num_scheduled_tokens,
            input_batch.query_start_loc_np,
            input_batch.prefill_len_np,
            input_batch.num_computed_prefill_tokens_np,
        )

        if not mm_embeds:
            # No MM tokens in this batch (e.g. all-decode step).
            # prepare_inputs will use embed_input_ids (text-only) directly.
            self._pending_mm_embeds = None
            return None

        # Stash raw MM ingredients for prepare_inputs to merge directly
        # into the persistent buffer, avoiding the intermediate copy
        # through encoder_runner.inputs_embeds.
        self._pending_mm_embeds = (mm_embeds, is_mm_embed)
        return None

    @torch.compile(dynamic=True)
    def _apply_self_conditioning(
        self,
        decode_slots: torch.Tensor,
        decode_idx: torch.Tensor,
        query_start_loc: torch.Tensor,
        inputs_embeds: torch.Tensor,
        sc_probs: torch.Tensor,
        is_encoder_phase: torch.Tensor,
        canvas_arange: torch.Tensor,
    ) -> None:
        starts = query_start_loc[decode_idx]
        token_indices = (canvas_arange + starts.unsqueeze(1)).reshape(-1)
        probs = sc_probs[decode_slots]
        probs = probs * (~is_encoder_phase[decode_slots]).unsqueeze(1).unsqueeze(2)
        probs = probs.reshape(-1, probs.size(-1))
        inputs_embeds[token_indices] = self.model.compute_self_conditioning(
            inputs_embeds[token_indices], probs
        )

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        states = self.diffusion_states
        num_tokens = input_batch.num_tokens
        num_reqs = input_batch.num_reqs

        # Write into the PERSISTENT inputs_embeds buffer so FULL CUDA graph
        # replay sees the latest values at the captured address.
        num_tokens_padded = input_batch.num_tokens_after_padding
        inputs_embeds = self._inputs_embeds_buf[:num_tokens_padded]

        # Populate embeddings: merge MM features when available,
        # otherwise embed input_ids as text-only.
        input_ids = input_batch.input_ids[:num_tokens]
        if self._pending_mm_embeds is not None:
            mm_embeds, is_mm_embed = self._pending_mm_embeds
            self._pending_mm_embeds = None
            inputs_embeds[:num_tokens].copy_(
                self.model.embed_input_ids(
                    input_ids,
                    multimodal_embeddings=mm_embeds,
                    is_multimodal=is_mm_embed,
                )
            )
        else:
            inputs_embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))

        # Apply self-conditioning ONLY for denoising decode requests.
        if input_batch.num_draft_tokens > 0 and self._req_id_to_index:
            slots_np = input_batch.idx_mapping_np[:num_reqs]
            num_logits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
            is_decode_indices_np = np.where(num_logits_np > 0)[0]
            num_decode = len(is_decode_indices_np)
            self.decode_slots.np[:num_decode] = slots_np[is_decode_indices_np]
            self.decode_slots.copy_to_uva()
            self.decode_idx.np[:num_decode] = is_decode_indices_np
            self.decode_idx.copy_to_uva()

            self._apply_self_conditioning(
                self.decode_slots.gpu[:num_decode],
                self.decode_idx.gpu[:num_decode],
                input_batch.query_start_loc,
                inputs_embeds,
                states.self_conditioning_probs,
                states.is_encoder_phase,
                self.canvas_arange,
            )

        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        # CUDA graph capture path — return a slice of the SAME persistent
        # inputs_embeds buffer that `prepare_inputs` writes to at runtime,
        # so the captured graph and runtime point to identical addresses.
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    def postprocess_state(self, idx_mapping, num_sampled) -> None:
        return None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
    ) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        max_query_len = input_batch.num_scheduled_tokens.max().item()

        # Per-request causal mode: encoder (commit) = causal,
        # denoise = bidirectional. Pass GPU tensor so the attention
        # backend can handle mixed batches.
        actual_num_reqs = input_batch.num_reqs
        slots = input_batch.idx_mapping[:actual_num_reqs]
        self._causal_buf[:actual_num_reqs] = self.diffusion_states.is_encoder_phase[
            slots
        ]
        if actual_num_reqs < num_reqs:
            self._causal_buf[actual_num_reqs:num_reqs] = False
        causal: bool | torch.Tensor = self._causal_buf[:num_reqs]

        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=causal,
        )

    num_new_sampled_tokens_per_step: int = 0


class DiffusionSampler:
    """Batched accept/renoise sampler for DiffusionGemma4.

    Follows the same structure as ``vllm.v1.worker.gpu.sample.sampler.Sampler``:
    decomposed into named methods, all GPU state in pre-allocated buffers,
    no GPU→CPU syncs on the hot path.
    """

    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: DiffusionGemma4RequestStates | None = None,
        renoise_ratio_modifier: float = 0.9,
        ar_mask_noise_proportion: float = 0.0,
        use_autoregressive_mask: bool = True,
        *,
        confidence_threshold: float,
        t_min: float,
        t_max: float,
        sampler_type: str,
        entropy_bound: float | None,
    ):
        self.sampler = sampler
        self.canvas_length = (
            diffusion_config.canvas_length if diffusion_config is not None else 32
        )
        self.t_min = t_min
        self.t_max = t_max
        self.confidence_threshold = confidence_threshold
        self.req_states = sampler.penalties_state.req_states
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.renoise_ratio_modifier = renoise_ratio_modifier
        self.ar_mask_noise_proportion = ar_mask_noise_proportion
        self.use_autoregressive_mask = use_autoregressive_mask
        self.use_entropy_bound = sampler_type == "entropy_bound"
        self.entropy_bound = entropy_bound
        self.sampling_states = sampler.sampling_states
        self.penalties_state = sampler.penalties_state

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        self._sampled = torch.zeros(
            max_num_reqs,
            self.canvas_length,
            dtype=torch.int32,
            device=device,
        )
        self._num_sampled = torch.zeros(
            max_num_reqs,
            dtype=torch.int32,
            device=device,
        )
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    def add_request(self, *args, **kwargs):
        self.sampler.add_request(*args, **kwargs)

    def apply_staged_writes(self):
        self.sampler.apply_staged_writes()

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------

    def _handle_prefill(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> SamplerOutput:
        states = self.diffusion_states
        num_reqs = input_batch.num_reqs
        CL = self.canvas_length
        slots = input_batch.idx_mapping[:num_reqs]
        states.init_canvas(slots)
        self.req_states.draft_tokens[slots, :CL] = states.canvas[slots]
        states.is_encoder_phase.index_fill_(0, slots.long(), False)
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_sampled,
        )

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    def _get_decode_requests(
        self,
        input_batch: Any,
        device: torch.device,
    ) -> tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor, int]:
        """Split batch into decode vs prefill, init canvas for new prefills."""
        states = self.diffusion_states
        CL = self.canvas_length
        num_reqs = input_batch.num_reqs
        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])

        decode_indices_np = np.where(per_req_nlogits_np > 0)[0]
        prefill_indices_np = np.where(per_req_nlogits_np == 0)[0]
        decode_slots_np = slots_np[decode_indices_np]

        if len(prefill_indices_np) > 0:
            ps = slots_np[prefill_indices_np]
            states.init_canvas(ps)
            self.req_states.draft_tokens[ps, :CL] = states.canvas[ps]
            ps_gpu = torch.from_numpy(ps.astype(np.int64)).to(
                states.is_encoder_phase.device
            )
            states.is_encoder_phase.index_fill_(0, ps_gpu, False)

        num_decode = len(decode_indices_np)
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]
        return per_req_nlogits_np, decode_slots_np, decode_slots, decode_idx, num_decode

    def _build_output(
        self,
        input_batch: Any,
        sampled: torch.Tensor,
        num_sampled: torch.Tensor,
        per_req_nlogits_np: np.ndarray,
        device: torch.device,
    ) -> SamplerOutput:
        """Compute num_rejected and build SamplerOutput."""
        num_reqs = input_batch.num_reqs

        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()

        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )

        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: torch.Tensor,
        input_batch: Any,
        draft_logits: torch.Tensor | None = None,
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device

        if input_batch.num_draft_tokens == 0:
            return self._handle_prefill(input_batch, device)

        # --- CPU/NumPy setup (outside compile) ---
        per_req_nlogits_np, _, decode_slots, decode_idx, num_decode = (
            self._get_decode_requests(input_batch, device)
        )

        sampled = self._sampled[:num_reqs]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()

        all_slots = input_batch.idx_mapping[:num_reqs]
        states = self.diffusion_states

        # --- Single compiled call: temp → sample → probs → post-process ---
        _compiled_sample_step(
            logits,
            decode_slots,
            decode_idx,
            all_slots,
            # State
            states.canvas,
            states.argmax_canvas,
            states.step,
            states.is_encoder_phase,
            states.confident,
            states.self_conditioning_probs,
            states.accepted_canvas_history,
            states.accepted_canvas_history_len,
            # Output
            sampled,
            num_sampled,
            self.req_states.draft_tokens,
            # Config
            max_denoising_steps=float(states.max_denoising_steps),
            t_min=self.t_min,
            t_max=self.t_max,
            confidence_threshold=self.confidence_threshold,
            vocab_size=self.vocab_size,
            CL=self.canvas_length,
            ST=states.stability_threshold,
            use_entropy_bound=self.use_entropy_bound,
            entropy_bound=self.entropy_bound or 0.0,
            renoise_ratio_modifier=self.renoise_ratio_modifier,
            ar_mask_noise_proportion=self.ar_mask_noise_proportion,
            use_autoregressive_mask=self.use_autoregressive_mask,
        )

        return self._build_output(
            input_batch, sampled, num_sampled, per_req_nlogits_np, device
        )

