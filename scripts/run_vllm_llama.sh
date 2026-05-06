#!/bin/bash

source /home1/xuh1010/miniconda3/etc/profile.d/conda.sh
conda activate vllm-server

MODEL_PATH="$HOME/models/Llama-3.1-8B-Instruct"

vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096

