# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run GSM8K eval via vLLM Python API.

Usage:
    python tests/v1/worker/run_gsm8k.py [--limit N] [--max-tokens N] [--max-num-seqs N]
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

MODEL_PATH = "/home/LucasWilkinson/local/test-checkpoint-26B-v2"

FEW_SHOT = (
    "Q: There are 15 trees in the grove. Grove workers will plant trees in "
    "the grove today. After they are done, there will be 21 trees. How many "
    "trees did the grove workers plant today?\n"
    "A: There are 15 trees originally. Then there were 21 trees after some "
    "more were planted. So there must have been 21 - 15 = 6. "
    "#### 6\n\n"
    "Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how "
    "many pieces do they have left in total?\n"
    "A: Originally, Leah had 32 chocolates. Her sister had 42. So in total "
    "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. "
    "#### 39\n\n"
)


def extract_answer(text: str) -> str:
    matches = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if matches:
        return matches[-1].replace(",", "")
    numbers = re.findall(r"-?\d[\d,]*", text)
    return numbers[-1].replace(",", "") if numbers else ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples (default: full test set)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=4)
    args = parser.parse_args()

    split = "test" if args.limit is None else f"test[:{args.limit}]"
    ds = load_dataset("openai/gsm8k", "main", split=split)

    llm = LLM(
        model=MODEL_PATH,
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
            f"<bos><|turn>user\n{FEW_SHOT}"
            f"Q: {row['question']}\nA:<turn|>\n<|turn>model\n"
        )
        ids = tok.encode(text, add_special_tokens=False)
        prompts.append(TokensPrompt(prompt_token_ids=ids))

    params = SamplingParams(max_tokens=args.max_tokens, temperature=0)

    t0 = time.time()
    outputs = llm.generate(prompts, params)
    elapsed = time.time() - t0

    correct = 0
    for i, (out, row) in enumerate(zip(outputs, ds)):
        text = out.outputs[0].text
        pred = extract_answer(text)
        gold = extract_answer(row["answer"])
        match = pred == gold
        correct += int(match)
        if not match:
            print(f"[{i:4d} FAIL] pred={pred:>8s} gold={gold:>8s}  {text[:60]}")

    n = len(ds)
    print(f"\nGSM8K: {correct}/{n} ({100 * correct / n:.1f}%)")
    print(f"Time: {elapsed:.0f}s ({elapsed / n:.2f}s/sample)")


if __name__ == "__main__":
    main()
