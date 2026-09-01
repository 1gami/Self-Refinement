#!/usr/bin/env bash
set -euo pipefail

: "${INPUT:?Set INPUT to the input JSONL path.}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}"
DATASET="${DATASET:-fever}"
TOP_K="${TOP_K:-3}"
THRESHOLD="${THRESHOLD:-0.3}"
MAX_ITER="${MAX_ITER:-3}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
EARLY_STOP="${EARLY_STOP:-1}"

export MODEL_PATH
MODEL_NAME="$(basename "$MODEL_PATH")"
OUTPUT_DIR="${OUTPUT_DIR:-results/${DATASET}}"
mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  --input "$INPUT"
  --dataset "$DATASET"
  --top-k "$TOP_K"
  --conf-threshold "$THRESHOLD"
  --num-iterations "$MAX_ITER"
  --overwrite
)

if [[ "$EARLY_STOP" == "1" ]]; then
  COMMON_ARGS+=(--early-stop)
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  COMMON_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "===== Run iterative WordR ====="
python run_experiment_wordr_iter.py \
  "${COMMON_ARGS[@]}" \
  --output "${OUTPUT_DIR}/${MODEL_NAME}_${DATASET}_wordr_iter${MAX_ITER}_k${TOP_K}_thr${THRESHOLD}.jsonl"

echo "===== Run iterative WordCand ====="
python run_experiment_word_cand_iter.py \
  "${COMMON_ARGS[@]}" \
  --output "${OUTPUT_DIR}/${MODEL_NAME}_${DATASET}_wordcand_iter${MAX_ITER}_k${TOP_K}_thr${THRESHOLD}.jsonl"

echo "===== Done ====="
