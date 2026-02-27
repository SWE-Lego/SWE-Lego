#!/bin/bash

set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Usage: $0 <input.jsonl> [output.jsonl]"
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="${2:-outputs/verifier_predictions_30b_a3b.jsonl}"

cd LLaMA-Factory-0.9.4.dev0
conda activate lf

python tts/inference_verifier_tts.py \
  --model_path "${MODEL_PATH:-SWE-Lego/SWE-Lego-Verifier-30B-A3B}" \
  --input_path "$INPUT_PATH" \
  --output_path "$OUTPUT_PATH" \
  --bf16
