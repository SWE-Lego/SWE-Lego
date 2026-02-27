#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <verifier_input.jsonl> [output.jsonl]"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LF_ROOT="${REPO_ROOT}/LLaMA-Factory-0.9.4.dev0"

DATA_PATH="$(realpath "$1")"
OUTPUT_PATH="${2:-${REPO_ROOT}/outputs/verifier_predictions_8b.jsonl}"
OUTPUT_PATH="$(realpath -m "${OUTPUT_PATH}")"

MODEL_PATH="${MODEL_PATH:-SWE-Lego/SWE-Lego-Verifier-8B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-131072}"
CLEAR_CACHE_STEPS="${CLEAR_CACHE_STEPS:-10}"
FLASH_ATTN="${FLASH_ATTN:-fa2}"
MASTER_PORT="${MASTER_PORT:-29501}"

if [ -z "${NUM_GPUS:-}" ]; then
  NUM_GPUS="$(nvidia-smi --list-gpus | wc -l)"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

cd "${LF_ROOT}"
conda activate lf

EXTRA_ARGS=()
if [ "${ENABLE_LIGER_KERNEL:-0}" = "1" ]; then
  EXTRA_ARGS+=(--enable_liger_kernel)
fi

torchrun \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  tts/inference_verifier_tts.py \
  --model_path "${MODEL_PATH}" \
  --data_path "${DATA_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --batch_size "${BATCH_SIZE}" \
  --max_length "${MAX_LENGTH}" \
  --clear_cache_steps "${CLEAR_CACHE_STEPS}" \
  --flash_attn "${FLASH_ATTN}" \
  --bf16 \
  "${EXTRA_ARGS[@]}"
