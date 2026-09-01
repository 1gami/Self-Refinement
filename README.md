# Self-Refinement for Fact Verification

This repository contains an experimental framework for evaluating **LLM self-refinement in three-way fact verification**. It compares direct prediction and several self-refinement baselines with two perturbation-based rationale methods, **WordR** and **WordCand**.

The supported verdict space is normalized to:

- `SUPPORT`
- `REFUTE`
- `NOT ENOUGH INFO`

The experiment code is designed to communicate with an **OpenAI-compatible endpoint**, with a local [vLLM](https://github.com/vllm-project/vllm) server as the default setup.

## Methods

| Method | Description |
| --- | --- |
| Direct | Predicts the verdict directly from the claim and evidence. |
| Basic Self-Refine | Generates generic self-feedback and revises the initial answer. |
| Evidence-Aware Self-Refine | Reviews the initial answer with explicit emphasis on evidence consistency before refinement. |
| VitaminC-Style Self-Refine | Uses contrast-oriented feedback before refinement. |
| WordR | Extracts candidate words/spans, masks each candidate, and verifies rationale importance using label flips or confidence drops. Top-ranked verified rationales are then used for refinement. |
| WordCand | Uses cue-based candidate spans such as negation, directionality, temporal, numerical, scope, and relation cues, then applies perturbation-based rationale verification before refinement. |

For WordR, spaCy NER/POS extraction is used when `en_core_web_sm` is available. Otherwise, the code automatically falls back to regex-based candidate extraction.

## Repository Structure

```text
Self-Refinement/
├── README.md
├── requirements.txt
├── llm_client.py
├── llm_client_wordr.py
├── prompts.py
├── prompts_original.py
├── prompts_wordr.py
├── prompts_word_cand.py
├── rationale_refine_wordr.py
├── rationale_refine_word_cand.py
├── run_experiment.py
├── run_experiment_sample5.py
├── run_experiment_wordr.py
├── run_experiment_wordr_iter.py
├── run_experiment_word_cand.py
├── run_experiment_word_cand_iter.py
├── run_grid_wordr.py
├── run_all.methods.sh
├── run_all.methods_iter.sh
├── examples/
│   └── example_input.jsonl
├── scripts/
│   ├── convert_scifact_unified.py
│   ├── run_thresholds.sh
│   ├── run_vllm_qwen.sh
│   ├── run_vllm_llama.sh
│   └── run_vllm_mistral.sh
└── analysis/
    ├── evaluation.py
    ├── extract_error_cases.py
    └── wandb_upload_eval.py
```

Additional analysis utilities (`compare_methods.py`, `reevaluate_normalized.py`, and `sum_ablation.py`) are retained at the repository root for compatibility with existing experiment outputs.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For spaCy-based candidate extraction:

```bash
python -m spacy download en_core_web_sm
```

The repository expects an OpenAI-compatible inference server. For local GPU inference, install vLLM according to the CUDA/PyTorch environment of the machine.

## Model Server

The default model identifier is `Qwen/Qwen2.5-7B-Instruct`, and the default endpoint is `http://127.0.0.1:8000/v1`.

Start a Qwen server:

```bash
bash scripts/run_vllm_qwen.sh
```

Alternative launchers are provided for Llama and Mistral:

```bash
bash scripts/run_vllm_llama.sh
bash scripts/run_vllm_mistral.sh
```

The launch scripts can be configured with environment variables. For example:

```bash
MODEL_PATH=/path/to/Qwen2.5-7B-Instruct \
VLLM_PORT=8000 \
bash scripts/run_vllm_qwen.sh
```

The experiment clients use the following variables:

```bash
export MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export OPENAI_API_KEY=EMPTY
```

When `MODEL_PATH` is changed for the vLLM server, use the same value for the experiment process so that the requested model name matches the served model name.

## Input Format

Experiment runners expect JSONL input with at least `claim`, `evidence`, and a gold-label field.

```json
{
  "id": "example_001",
  "claim": "The treatment reduced mortality.",
  "evidence": "Mortality was lower in the treatment group than in the control group.",
  "label": "SUPPORT"
}
```

Required fields:

| Field | Description |
| --- | --- |
| `claim` | Claim to verify. |
| `evidence` | Evidence provided to the model. |
| `label` | Gold verdict. `verified_label` is used instead when present and non-empty. |

Optional metadata such as `id`, `source`, `need_review`, and `comment` is preserved when available.

A small synthetic example file is included at `examples/example_input.jsonl` for checking the input format.

## Running Experiments

### 1. Baseline self-refinement

Runs direct prediction, Basic Self-Refine, Evidence-Aware Self-Refine, and VitaminC-Style Self-Refine.

```bash
python run_experiment.py \
  --input examples/example_input.jsonl \
  --dataset example \
  --output results/example/baselines.jsonl \
  --overwrite
```

### 2. WordR

```bash
python run_experiment_wordr.py \
  --input examples/example_input.jsonl \
  --dataset example \
  --top-k 5 \
  --conf-threshold 0.2 \
  --output results/example/wordr.jsonl \
  --overwrite
```

Default WordR hyperparameters:

- `top-k = 5`
- `confidence threshold = 0.2`

### 3. WordCand

```bash
python run_experiment_word_cand.py \
  --input examples/example_input.jsonl \
  --dataset example \
  --top-k 5 \
  --conf-threshold 0.2 \
  --output results/example/wordcand.jsonl \
  --overwrite
```

### 4. Iterative refinement

The iterative runners support multiple refinement iterations and optional early stopping.

```bash
python run_experiment_wordr_iter.py \
  --input examples/example_input.jsonl \
  --dataset example \
  --top-k 5 \
  --conf-threshold 0.2 \
  --num-iterations 3 \
  --early-stop \
  --output results/example/wordr_iter.jsonl \
  --overwrite
```

The corresponding WordCand runner is `run_experiment_word_cand_iter.py`.

### 5. Run WordR and WordCand together

`run_all.methods.sh` reads its configuration from environment variables.

```bash
INPUT=examples/example_input.jsonl \
DATASET=example \
TOP_K=5 \
THRESHOLD=0.2 \
bash run_all.methods.sh
```

For iterative runs:

```bash
INPUT=examples/example_input.jsonl \
DATASET=example \
TOP_K=5 \
THRESHOLD=0.2 \
MAX_ITER=3 \
bash run_all.methods_iter.sh
```

## Threshold Search

A WordR grid runner is provided for `top-k` and confidence-drop thresholds.

```bash
python run_grid_wordr.py \
  --input examples/example_input.jsonl \
  --dataset example \
  --top-k-list 3 5 7 \
  --threshold-list 0.1 0.2 0.3 \
  --overwrite
```

The convenience shell script evaluates thresholds from `0.10` to `0.30` at a fixed `TOP_K` value:

```bash
TOP_K=5 bash scripts/run_thresholds.sh examples/example_input.jsonl example
```

## SciFact Conversion

`convert_scifact_unified.py` converts SciFact claims and corpus files into the unified JSONL format used by the experiment runners.

```bash
python scripts/convert_scifact_unified.py \
  --claims /path/to/claims_dev.jsonl \
  --corpus /path/to/corpus.jsonl \
  --output data/scifact/scifact_dev.jsonl \
  --split dev \
  --evidence-mode gold_only
```

Use `--evidence-mode full_abstract` to construct evidence from the full cited abstract. Use `--no-nei` to exclude automatically constructed NEI examples with empty gold evidence.

## Evaluation

Evaluate an experiment result file with:

```bash
python analysis/evaluation.py \
  --input results/example/wordr.jsonl \
  --out-dir results/evaluation \
  --save-error-analysis
```

The evaluator reports classification metrics, confusion matrices, refinement transitions, and WordR rationale statistics, and writes a JSON summary to the output directory.

More targeted error subsets can be extracted with:

```bash
python analysis/extract_error_cases.py \
  --input results/example/wordr.jsonl \
  --out-dir results/error_cases
```

Evaluation summaries can optionally be uploaded to Weights & Biases using `analysis/wandb_upload_eval.py`.

## Reproducibility Notes

- The experiment runners use deterministic decoding by default (`temperature=0.0`).
- Model outputs are normalized to a common three-class verdict space.
- WordR/WordCand require token log probabilities from the inference server.
- Results, logs, local datasets, model weights, and W&B artifacts are intentionally excluded from Git version control.
- Large benchmark datasets are not distributed in this repository; prepare them locally in the unified JSONL format described above.

## Acknowledgements

This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - [http://dsplus.uos.ac.kr/](http://dsplus.uos.ac.kr/)
