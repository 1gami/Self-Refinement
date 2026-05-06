#!/bin/bash
set -e

export MODEL_PATH=/home1/xuh1010/models/Qwen2.5-7B-Instruct

INPUT="data/sample/fever_3class_review_sample_900.jsonl"
DATASET="fever"
TOP_K=3
THRESHOLD=0.3
MAX_ITER=3

mkdir -p results/${DATASET}

echo "===== Run WordR with Early Stop ====="
python run_experiment_wordr_iter.py \
  --input ${INPUT} \
  --dataset ${DATASET} \
  --top-k ${TOP_K} \
  --conf-threshold ${THRESHOLD} \
  --num-iterations ${MAX_ITER} \
  --early-stop \
  --output results/${DATASET}/Last1_${DATASET}_wordr_earlystop_maxiter${MAX_ITER}_k${TOP_K}_thr${THRESHOLD}.jsonl \
  --overwrite

echo "===== Run WordCand with Early Stop ====="
python run_experiment_word_cand_iter.py \
  --input ${INPUT} \
  --dataset ${DATASET} \
  --top-k ${TOP_K} \
  --conf-threshold ${THRESHOLD} \
  --num-iterations ${MAX_ITER} \
  --early-stop \
  --output results/${DATASET}/Last1_${DATASET}_wordcand_earlystop_maxiter${MAX_ITER}_k${TOP_K}_thr${THRESHOLD}.jsonl \
  --overwrite

echo "===== Done ====="