# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Smoke test: load a (possibly quantized) DiffusionGemma checkpoint and run a
handful of GSM8K samples to confirm it loads and generates sane output.

Usage:
    python run_quant_smoke.py <model_path_or_repo_id> [--limit N]
"""

import argparse
import os
import sys
import time

import regex as re

sys.path.insert(0, "/home/LucasWilkinson/local/transformers-gdm/src")
os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"

from datasets import load_dataset

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

FEW_SHOT = (
    "Q: There are 15 trees in the grove. Grove workers will plant trees in "
    "the grove today. After they are done, there will be 21 trees. How many "
    "trees did the grove workers plant today?\n"
    "A: There are 15 trees originally. Then there were 21 trees after some "
    "more were planted. So there must have been 21 - 15 = 6. #### 6\n\n"
)


def extract(text: str) -> str:
    m = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if m:
        return m[-1].replace(",", "")
    nums = re.findall(r"-?\d[\d,]*", text)
    return nums[-1].replace(",", "") if nums else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--limit", type=int, default=4)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.limit}]")
    llm = LLM(
        model=args.model,
        diffusion_config={"canvas_length": 256},
        max_model_len=1024,
        max_num_seqs=args.max_num_seqs,
        trust_remote_code=True,
        attention_backend="TRITON_ATTN",
    )
    tok = llm.get_tokenizer()
    prompts = []
    for row in ds:
        text = (
            f"<bos><|turn>user\n{FEW_SHOT}Q: {row['question']}\nA:<turn|>\n"
            f"<|turn>model\n"
        )
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    params = SamplingParams(max_tokens=256, temperature=0)
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0

    correct = 0
    for out, row in zip(outs, ds):
        text = out.outputs[0].text
        ok = extract(text) == extract(row["answer"])
        correct += int(ok)
        print(f"[{'OK ' if ok else 'XX '}] pred={extract(text):>8s} "
              f"gold={extract(row['answer']):>8s} :: {text[:70]!r}")
    n = len(ds)
    print(f"\nSMOKE: {correct}/{n} correct, {dt:.0f}s")


if __name__ == "__main__":
    main()
