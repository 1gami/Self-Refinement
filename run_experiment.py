#-----------baseline 전용 실행 파일임 -----------------
# --------------------MODEL------------------------
# direct
# basic_self_refine
# evidence_aware_self_refine
# vitaminc_style_self_refine

# -------------------실행 방법------------------------
# - llm_client.py 사용
# - logprobs / confidence 사용 안 함
# - word-level rationale 없음
# - 가장 기본 self-refinement baseline 결과 저장
# --------------------------------------------------

import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter

from llm_client import call_llm
from prompts import (
    direct_prompt,
    basic_feedback_prompt,
    basic_refine_prompt,
    evidence_aware_feedback_prompt,
    evidence_aware_refine_prompt,
    vitaminc_style_feedback_prompt,
    vitaminc_style_refine_prompt,
)


def read_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            data.append(json.loads(line))

    return data


def normalize_label(label):
    """Unify FEVER/SciFact/VitaminC label variants to 3-class labels."""
    if label is None:
        return "PARSE_ERROR"

    x = str(label).strip().upper().replace("_", " ")

    if x in {"SUPPORT", "SUPPORTS", "SUPPORTED", "ENTAILMENT", "TRUE"}:
        return "SUPPORT"

    if x in {
        "REFUTE", "REFUTES", "REFUTED",
        "CONTRADICT", "CONTRADICTS", "CONTRADICTION", "FALSE"
    }:
        return "REFUTE"

    if x in {"NOT ENOUGH INFO", "NEI", "NOT ENOUGH INFORMATION", "INSUFFICIENT INFO"}:
        return "NOT ENOUGH INFO"

    return "PARSE_ERROR"


def parse_label(output):
    # Handles outputs like: Label: SUPPORTS, Label: [REFUTES], Label: CONTRADICT
    match = re.search(
        r"Label:\s*\[?\s*"
        r"(SUPPORTS?|SUPPORTED|REFUTES?|REFUTED|CONTRADICTS?|CONTRADICTION|NOT ENOUGH INFO|NEI)"
        r"\s*\]?",
        output,
        re.IGNORECASE,
    )

    if match:
        return normalize_label(match.group(1))

    # Fallback: if the model omitted 'Label:' but included a label word.
    return normalize_label(output)


def get_gold_label(sample):
    """Use verified_label if available; otherwise use label. Always normalize."""
    verified_label = sample.get("verified_label")
    if verified_label is not None and str(verified_label).strip():
        return normalize_label(verified_label)
    return normalize_label(sample.get("label"))


def get_model_short_name():
    model_path = os.environ.get("MODEL_PATH", "qwen")
    return Path(model_path).name.replace("/", "_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL with claim/evidence/label fields")
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g., fever/scifact/vitaminc")
    parser.add_argument("--output", default=None, help="Output result JSONL path")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    args = parser.parse_args()

    input_path = args.input
    dataset_name = args.dataset or Path(input_path).parent.name

    samples = read_jsonl(input_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]

    print(f"Loaded {len(samples)} samples from {input_path}")
    print("Gold label distribution:", Counter(get_gold_label(ex) for ex in samples))

    results = []

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", f"sample_{idx}")

        print(f"\n===== Running sample {idx}/{len(samples)} | id={sample_id} =====", flush=True)

        claim = sample["claim"]
        evidence = sample["evidence"]
        gold_label = get_gold_label(sample)

        # direct prediction = self-refine의 initial answer와 동일
        direct = direct_prediction(claim, evidence)
        initial_answer = direct["answer"]

        print(f"  [Gold] {gold_label}", flush=True)
        print(f"  [Direct] {direct['label']}", flush=True)

        row = {
            "id": sample_id,
            "dataset": dataset_name,
            "claim": claim,
            "evidence": evidence,
            "gold_label": gold_label,
            "source": sample.get("source", ""),
            "need_review": sample.get("need_review", None),
            "comment": sample.get("comment", ""),
            "direct": direct,
        }

        row["basic_self_refine"] = basic_self_refinement(
            claim,
            evidence,
            initial_answer,
        )
        print(
            f"  [Basic Self-Refine] {row['basic_self_refine']['refined_label']}",
            flush=True,
        )

        row["evidence_aware_self_refine"] = evidence_aware_self_refinement(
            claim,
            evidence,
            initial_answer,
        )
        print(
            f"  [Evidence-aware] {row['evidence_aware_self_refine']['refined_label']}",
            flush=True,
        )

        row["vitaminc_style_self_refine"] = vitaminc_style_self_refinement(
            claim,
            evidence,
            initial_answer,
        )
        print(
            f"  [VitaminC-style] {row['vitaminc_style_self_refine']['refined_label']}",
            flush=True,
        )

        results.append(row)

    print("\n===== Accuracy =====")
    methods = [
        "direct",
        "basic_self_refine",
        "evidence_aware_self_refine",
        "vitaminc_style_self_refine",
    ]

    for method in methods:
        acc = evaluate(results, method)
        pred_dist = label_counter(results, method)
        print(f"  {method}: {acc:.3f} | pred_dist={dict(pred_dist)}")

    model_name = get_model_short_name()
    out_path = Path(args.output) if args.output else Path("results") / dataset_name / f"{model_name}_{dataset_name}_baselines.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()