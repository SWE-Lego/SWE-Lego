#!/usr/bin/env bash

cd OpenHands-0.53.0
conda activate openhands

YOUR_OUTPUT_JSONL=evaluation/evaluation_outputs/outputs/princeton-nlp__SWE-bench_Verified-test/CodeActAgent/Qwen3-32B_maxiter_100_N_v0.53.0-no-hint-noplan-run_1/output.jsonl
DATASET=princeton-nlp/SWE-bench_Verified
DATASET_SPLIT=test

./evaluation/benchmarks/swe_bench/scripts/eval_infer.sh $YOUR_OUTPUT_JSONL  dataset_name=$DATASET split=$DATASET_SPLIT --convert-only


############################################################################################################################################################################################

cd ../SWE-bench-4.0.4
conda activate swebench

YOUR_OUTPUT_JSONL1=../OpenHands-0.53.0/$YOUR_OUTPUT_JSONL

jsonl_path=${YOUR_OUTPUT_JSONL1/output.jsonl/output.swebench.jsonl}

python -m swebench.harness.run_evaluation \
    --max_workers 10 \
    --dataset_name $DATASET \
    --report_dir ./results/ \
    --cache_level instance \
    --predictions_path $jsonl_path \
    --run_id openhands \
    --timeout 500