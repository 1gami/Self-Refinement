#!/bin/bash
set -e

export MODEL_PATH=/home1/xuh1010/models/Qwen2.5-7B-Instruct

INPUT="data/sample/fever_3class_review_sample_99.jsonl"
DATASET="fever"
TOP_K=3
THRESHOLD=0.3
# MAX_SAMPLES=100

MODEL_NAME=$(basename "$MODEL_PATH")

mkdir -p results/${DATASET}

echo "===== Run WordR ====="
python run_experiment_wordr.py \
  --input ${INPUT} \
  --dataset ${DATASET} \
  --top-k ${TOP_K} \
  --conf-threshold ${THRESHOLD} \
  # --max-samples ${MAX_SAMPLES} \
  --output results/${DATASET}/${MODEL_NAME}_${DATASET}_WordRationale${TOP_K}_thr${THRESHOLD}.jsonl \
  --overwrite

echo "===== Run WordCand ====="
python run_experiment_word_cand.py \
  --input ${INPUT} \
  --dataset ${DATASET} \
  --top-k ${TOP_K} \
  --conf-threshold ${THRESHOLD} \
  # --max-samples ${MAX_SAMPLES} \
  --output results/${DATASET}/${MODEL_NAME}_${DATASET}_WordCandidate${TOP_K}_thr${THRESHOLD}.jsonl \
  --overwrite

echo "===== Done ====="