# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AIME2025 eval via inspect_ai + vLLM diffusion."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, "/home/LucasWilkinson/local/transformers-gdm/src")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from inspect_ai import Epochs, eval
from inspect_evals.aime2025 import aime2025

MODEL_PATH = "/home/LucasWilkinson/local/test-checkpoint-26B-v2"


def main():
    logs = eval(
        aime2025(),
        epochs=Epochs(4, reducer="pass_at_4"),
        model=f"vllm/{MODEL_PATH}",
        model_args={
            "diffusion_config": '{"canvas_length": 256}',
            "max_model_len": 8192,
            "max_num_seqs": 1,
            "trust_remote_code": True,
            "attention_backend": "TRITON_ATTN",
        },
        max_tokens=4096,
        display="plain",
    )
    print(logs)


if __name__ == "__main__":
    main()
