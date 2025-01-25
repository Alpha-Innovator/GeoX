#!/bin/bash

MODEL_BASE_PATH="${1:-experiments/geox_pgps9k}"
QUESTION_FILE="${2:-data/pgps9k/question.jsonl}"
IMAGE_FOLDER="${3:-data/pgps9k/images}"

MODEL_PATHS=("$MODEL_BASE_PATH" $(ls -d "$MODEL_BASE_PATH"/*/ | sed 's:/*$::'))


for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    OUTPUT_FILE="${MODEL_PATH%/}/result.jsonl"

    echo "Running inference on model: $MODEL_PATH"
    echo "Results will be saved to: $OUTPUT_FILE"

    python eval/inference.py \
        --model-path "$MODEL_PATH" \
        --question-file "$QUESTION_FILE" \
        --image-folder "$IMAGE_FOLDER" \
        --answers-file "$OUTPUT_FILE" \
        --conv-mode "geo_v1"

    echo "Scoring ... $OUTPUT_FILE"
    python eval/score_geometry3k.py \
        --pred_file "$OUTPUT_FILE" \
        --gt_file "./data/pgps9k/test.json"

    echo "----------------------------------------"

done

