#!/usr/bin/env bash

cd OpenHands-0.53.0
conda activate openhands


MODEL_CONFIG=llm.eval_qwen3_8b
GIT_VERSION=HEAD
AGENT=CodeActAgent  # 默认CodeActAgent
EVAL_LIMIT=500  # 推理数量
MAX_ITER=100  # 每个样本的最大交互轮数, 默认100
NUM_WORKERS=24  # 并发工作进程数
DATASET=princeton-nlp/SWE-bench_Verified
DATASET_SPLIT=test  # 默认test
N_RUNS=1    # 重复运行次数，默认1
MODE=swe # e.g. swt, swt-ci, or swe. 默认swe.

# Use hint text in the evaluation (default: false)
export USE_HINT_TEXT=false # Ignore this if you are not sure.

# Specify a condenser configuration for memory management (default: NoOpCondenser)
# export EVAL_CONDENSER=recent_for_eval # Name of the condenser config group in config.toml
# export EVAL_CONDENSER=summarizer_for_eval # Name of the condenser config group in config.toml

# Specify the instruction prompt template file name
export INSTRUCTION_TEMPLATE_NAME=swe_default.j2 # Name of the file in the swe_bench/prompts folder.

# should be true
export ENABLE_PLAN_MODE=false

export ADD_IN_CONTEXT_LEARNING_EXAMPLE=false

./evaluation/benchmarks/swe_bench/scripts/run_infer.sh $MODEL_CONFIG $GIT_VERSION $AGENT $EVAL_LIMIT $MAX_ITER $NUM_WORKERS $DATASET $DATASET_SPLIT $N_RUNS $MODE
