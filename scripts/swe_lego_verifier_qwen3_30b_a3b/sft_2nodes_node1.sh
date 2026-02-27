#!/usr/bin/env bash

set -euo pipefail

cd LLaMA-Factory-0.9.4.dev0
conda activate lf

export WANDB_API_KEY=${WANDB_API_KEY:-<YOUR_WANDB_KEY>}

MASTER_ADDR=${MASTER_ADDR:?Please set MASTER_ADDR to node0 hostname/IP}
MASTER_PORT=${MASTER_PORT:-20812}
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

torchrun \
  --nnodes "${NNODES}" \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  --node_rank 1 \
  src/train.py examples/train_full/swe_lego_verifier_qwen3_30b_a3b_18k_2nodes.yaml
