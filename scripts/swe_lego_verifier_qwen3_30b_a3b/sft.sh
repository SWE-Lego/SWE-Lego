#!/usr/bin/env bash

set -euo pipefail

cd LLaMA-Factory-0.9.4.dev0
conda activate lf

export WANDB_API_KEY=${WANDB_API_KEY:-<YOUR_WANDB_KEY>}

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/swe_lego_verifier_qwen3_30b_a3b_18k.yaml
