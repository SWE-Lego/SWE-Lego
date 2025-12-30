#!/bin/bash

cd LLaMA-Factory-0.9.4.dev0
conda activate lf

export WANDB_API_KEY=<YOUR_WANDB_KEY>

FORCE_TORCHRUN=1 llamafactory-cli train examples/train_full/swe_lego_qwen3_8b.yaml