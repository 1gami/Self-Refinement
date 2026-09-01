#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input.jsonl> [dataset]" >&2
  exit 2
fi

INPUT="$1"
DATASET="${2:-fever}"
TOP_K="${TOP_K:-5}"
OUT_DIR="${OUT_DIR:-results/grid_wordr}"

python run_grid_wordr.py \
  --input "$INPUT" \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --top-k-list "$TOP_K" \
  --threshold-list 0.10 0.15 0.20 0.25 0.30 \
  --overwrite
