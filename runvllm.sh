#!/bin/bash
#    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
#  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
podman run --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8000:8000 \
  --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model mistralai/Mistral-7B-v0.1
