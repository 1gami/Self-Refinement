# Structure
self_refine_exp/
├── data/
│   ├── sample/
│   └── scifact/
│       ├── scifact_dev.jsonl
│       └── scifact_train.jsonl
│
├── results/
│   ├── fever/
│   ├── scifact/
│   └── thresholds/
│
├── scripts/
│   ├── convert_scifact_unified.py
│   ├── run_vllm_qwen.sh
│   ├── run_vllm_llama.sh
│   ├── run_vllm_mistral.sh
│   └── run_thresholds.sh
│
├── analysis/
│   ├── evaluation.py
│   ├── extract_error_cases.py
│   └── wandb_upload_eval.py
│
├── logs/
│   ├── vllm/
│   └── wordr/
│
├── wandb/
│
├── llm_client.py
├── llm_client_wordr.py
├── prompts.py
├── prompts_wordr.py
├── prompts_word_cand.py
├── prompts_ablation.py
├── rationale_refine_wordr.py
├── rationale_refine_word_cand.py
├── run_experiment.py
├── run_experiment_wordr.py
├── run_experiment_word_cand.py
└── run_experiment_sample5.py


python run_wordr.py \
  --input data/scifact/scifact_dev.jsonl \
  --dataset scifact \
  --output results/scifact/qwen_scifact_wordr.jsonl \
  --overwrite

python run_wordr.py \
  --input data/scifact/scifact_dev.jsonl \
  --dataset scifact \
  --top-k 5 \
  --conf-threshold 0.2 \
  --overwrite

python run_wordcand.py \
  --input data/scifact/scifact_dev.jsonl \
  --dataset scifact \
  --top-k 5 \
  --conf-threshold 0.2 \
  --overwrite

Default top-k : 5 , confidence threshold : 0.2

# FEVER Baseline
python run_baselines.py \
  --input data/fever_3class_review_sample.jsonl \
  --dataset fever \
  --output results/fever/qwen_fever_baselines.jsonl \
  --overwrite

# FEVER Word-level rationale
python run_wordr.py \
  --input data/fever_3class_review_sample.jsonl \
  --dataset fever \
  --output results/fever/qwen_fever_wordr.jsonl \
  --overwrite

# SciFact Word-level rationale
python run_wordr.py \
  --input data/scifact/scifact_dev.jsonl \
  --dataset scifact \
  --output results/scifact/qwen_scifact_wordr.jsonl \
  --overwrite

# SciFact Word-level Candidate rationale
python run_wordcand.py \
  --input data/scifact/scifact_dev.jsonl \
  --dataset scifact \
  --output results/scifact/qwen_scifact_wordcand.jsonl \
  --overwrite