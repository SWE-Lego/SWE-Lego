#!/usr/bin/env bash

PROCESS_FILEPATH=$1
if [ -z "$PROCESS_FILEPATH" ]; then
    echo "Error: PROCESS_FILEPATH is empty. Usage: ./eval_infer.sh <output_file> [instance_id] [dataset_name] [split]"
    exit 1
fi

if [ ! -f $PROCESS_FILEPATH ]; then
    echo "Error: $PROCESS_FILEPATH is not a file"
    exit 1
fi

# If instance_id is empty, it means we want to eval on the whole $PROCESS_FILEPATH
# otherwise, we want to eval on the instance_id
INSTANCE_ID=$2
DATASET_NAME=${3:-"princeton-nlp/SWE-bench_Lite"}
SPLIT=${4:-"test"}
ENVIRONMENT=${5:-"local"}

echo "INSTANCE_ID: $INSTANCE_ID"
echo "DATASET_NAME: $DATASET_NAME"
echo "SPLIT: $SPLIT"

if [[ "$ENVIRONMENT" != "local" && "$ENVIRONMENT" != "modal" ]]; then
    echo "Error: ENVIRONMENT must be either 'local' or 'modal'"
    exit 1
fi

echo "ENVIRONMENT: $ENVIRONMENT"

PROCESS_FILEPATH=$(realpath $PROCESS_FILEPATH)
FILE_DIR=$(dirname $PROCESS_FILEPATH)
FILE_NAME=$(basename $PROCESS_FILEPATH)

echo "Evaluating $FILE_NAME @ $FILE_DIR"

# ================================================
# detect whether PROCESS_FILEPATH is in OH format or in SWE-bench format
echo "=============================================================="
echo "Detecting whether PROCESS_FILEPATH is in OH format or in SWE-bench format"
echo "=============================================================="
# SWE-bench format is a JSONL where every line has three fields: model_name_or_path, instance_id, and model_patch
function is_swebench_format() {
    # Read the first line of the file
    read -r first_line < "$PROCESS_FILEPATH"

    # Use jq to check if the first line has the required fields
    echo "$first_line" | jq -e '. | has("model_name_or_path") and has("instance_id") and has("model_patch")' > /dev/null

    if [ $? -ne 0 ]; then
        return 1 # Return 1 if the first line does not have the required fields
    fi

    return 0 # Return 0 if the first line has the required fields
}
# Call the function with the file path
is_swebench_format "$PROCESS_FILEPATH"
IS_SWEBENCH_FORMAT=$?
# Use the result in an if-else statement
if [ $IS_SWEBENCH_FORMAT -eq 0 ]; then
    echo "The file IS in SWE-bench format."
    SWEBENCH_FORMAT_JSONL=$PROCESS_FILEPATH
else
    echo "The file IS NOT in SWE-bench format."

    # ==== Convert OH format to SWE-bench format ====
    echo "Merged output file with fine-grained report will be saved to $FILE_DIR"
    poetry run python3 evaluation/benchmarks/swe_bench/scripts/eval/convert_oh_output_to_swe_json.py $PROCESS_FILEPATH
    # replace .jsonl with .swebench.jsonl in filename
    SWEBENCH_FORMAT_JSONL=${PROCESS_FILEPATH/.jsonl/.swebench.jsonl}
    echo "SWEBENCH_FORMAT_JSONL: $SWEBENCH_FORMAT_JSONL"
    # assert that the file exists
    if [ ! -f $SWEBENCH_FORMAT_JSONL ]; then
        echo "Error: $SWEBENCH_FORMAT_JSONL does not exist. There is probably an error in the conversion process."
        exit 1
    fi
    SWEBENCH_FORMAT_JSONL=$(realpath $SWEBENCH_FORMAT_JSONL)
fi
# ================================================
