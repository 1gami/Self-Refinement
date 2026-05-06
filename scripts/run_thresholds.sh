#!/bin/bash

THRESHOLDS=(0.1 0.15 0.2 0.25 0.3)

for TH in "${THRESHOLDS[@]}"
do
    echo "Running threshold=${TH}"

    python run_experiment_wordr.py \
        --confidence_drop_threshold ${TH} \
        --top_k 5
done