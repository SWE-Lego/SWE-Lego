#!/bin/bash

conda activate vllm

python -m vllm.entrypoints.openai.api_server \
    --model SWE-Lego/SWE-Lego-Qwen3-8B \
    --served-model-name Qwen/Qwen3-8B \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 163840 \
    --max-num-seqs 24 \
    --api-key "dummy-key"
