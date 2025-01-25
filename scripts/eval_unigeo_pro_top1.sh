#!/bin/bash

MODEL_BASE_PATH="${1:-experiments/geox_unigeo}"
QUESTION_FILE="${2:-data/unigeo/proving_question.jsonl}"
IMAGE_FOLDER="${3:-data/unigeo/images}"

MODEL_PATHS=("$MODEL_BASE_PATH" $(ls -d "$MODEL_BASE_PATH"/*/ | sed 's:/*$::'))


for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUTPUT_FILE="${MODEL_PATH%/}/result_pro_top1.jsonl"

    echo "Running inference on model: $MODEL_PATH"
    echo "Results will be saved to: $OUTPUT_FILE"

    python eval/inference.py \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$OUTPUT_FILE" \
        --conv-mode "geo_v1" \
        --num_return_sequences 1 \
        --max_new_tokens 128
    echo "----------------------------------------"

done
