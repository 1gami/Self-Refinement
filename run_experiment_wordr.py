#-----------baseline + Word-level Rationale 전용 실행 파일임 -----------------
# --------------------MODEL------------------------
# direct
# basic_self_refine
# evidence_aware_self_refine
# vitaminc_style_self_refine
# wordr_self_refine  ← Word-level Perturbation 기반
# -------------------실행 방법------------------------
# - llm_client_wordr.py 사용
# - call_llm_with_logprobs 사용
# - direct 단계에서 initial_conf 저장
# - rationale_refine_wordr.py의 wordr_self_refinement 호출
# - WordR 후보/검증 통계 출력
# -----------------------------------------------------
#  NER/POS 또는 regex로 후보 단어 추출
# → 각 후보 masking
# → label flip / confidence drop 확인
# → verified rationale top-k를 refinement prompt에 넣음
# --------------------------------------------------


import argparse
import json
import os
import re
from pathlib import Path
from collections import Counter

# ── 클라이언트: logprobs 지원 버전 사용 ──────────────────────
from llm_client_wordr import call_llm, call_llm_with_logprobs

# ── 기존 방식 프롬프트 (변경 없음) ───────────────────────────
from prompts import (
    direct_prompt,
    basic_feedback_prompt,
    basic_refine_prompt,
    evidence_aware_feedback_prompt,
    evidence_aware_refine_prompt,
    vitaminc_style_feedback_prompt,
    vitaminc_style_refine_prompt,
)

# ── 새 방식 ──────────────────────────────────────────────────
from rationale_refine_wordr import wordr_self_refinement


# ══════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════

RATIONALE_TOP_K           = 5
CONFIDENCE_DROP_THRESHOLD = 0.2   # ablation: 0.1 / 0.15 / 0.2 / 0.25 / 0.3


# ══════════════════════════════════════════════════════════════
# Helpers (기존과 동일)
# ══════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════
# Prediction 함수들
# ══════════════════════════════════════════════════════════════

def direct_prediction_with_conf(claim, evidence):
    """
    direct prediction + label confidence 동시 추출.
    wordr_self_refinement에 initial_conf 전달용.
    """
    prompt = direct_prompt(claim, evidence)
    answer, label_conf, label_token, _ = call_llm_with_logprobs(prompt)
    return {
        "answer":      answer,
        "label":       parse_label(answer),
        "label_conf":  round(label_conf, 4) if label_conf is not None else None,
        "label_token": label_token,
    }


def basic_self_refinement(claim, evidence, initial_answer, num_iterations=1):
    current_answer = initial_answer
    history = []

    for it in range(1, num_iterations + 1):
        feedback = call_llm(
            basic_feedback_prompt(claim, evidence, current_answer)
        )

        refined_answer = call_llm(
            basic_refine_prompt(claim, evidence, current_answer, feedback)
        )

        history.append({
            "iteration": it,
            "input_answer": current_answer,
            "feedback": feedback,
            "refined_answer": refined_answer,
            "refined_label": parse_label(refined_answer),
        })

        current_answer = refined_answer

    return {
        "initial_answer": initial_answer,
        "feedback": history[-1]["feedback"] if history else "",
        "refined_answer": current_answer,
        "initial_label": parse_label(initial_answer),
        "refined_label": parse_label(current_answer),
        "num_iterations": num_iterations,
        "iteration_history": history,
    }

def evidence_aware_self_refinement(claim, evidence, initial_answer):
    feedback       = call_llm(evidence_aware_feedback_prompt(claim, evidence, initial_answer))
    refined_answer = call_llm(evidence_aware_refine_prompt(claim, evidence, initial_answer, feedback))
    return {
        "initial_answer": initial_answer,
        "feedback":       feedback,
        "refined_answer": refined_answer,
        "initial_label":  parse_label(initial_answer),
        "refined_label":  parse_label(refined_answer),
    }


def vitaminc_style_self_refinement(claim, evidence, initial_answer):
    feedback       = call_llm(vitaminc_style_feedback_prompt(claim, evidence, initial_answer))
    refined_answer = call_llm(vitaminc_style_refine_prompt(claim, evidence, initial_answer, feedback))
    return {
        "initial_answer": initial_answer,
        "feedback":       feedback,
        "refined_answer": refined_answer,
        "initial_label":  parse_label(initial_answer),
        "refined_label":  parse_label(refined_answer),
    }


# ══════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════

def evaluate(results, method_name):
    correct = 0
    total   = 0

    for row in results:
        gold = normalize_label(row["gold_label"])

        if method_name == "direct":
            pred = normalize_label(row["direct"]["label"])
        else:
            pred = normalize_label(row[method_name]["refined_label"])

        if pred == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def label_counter(results, method_name):
    preds = []

    for row in results:
        if method_name == "direct":
            preds.append(normalize_label(row["direct"]["label"]))
        else:
            preds.append(normalize_label(row[method_name]["refined_label"]))

    return Counter(preds)


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL with claim/evidence/label fields")
    parser.add_argument("--dataset", default=None, help="Dataset name, e.g., fever/scifact/vitaminc")
    parser.add_argument("--output", default=None, help="Output result JSONL path")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it already exists")
    parser.add_argument("--top-k", type=int, default=RATIONALE_TOP_K)
    parser.add_argument("--conf-threshold", type=float, default=CONFIDENCE_DROP_THRESHOLD)
    args = parser.parse_args()

    input_path = args.input
    dataset_name = args.dataset or Path(input_path).parent.name

    samples = read_jsonl(input_path)
    if args.max_samples is not None:
        samples = samples[:args.max_samples]
    print(f"Loaded {len(samples)} samples from {input_path}")
    print("Gold label distribution:", Counter(get_gold_label(ex) for ex in samples))
    print(f"[WordR] top_k={args.top_k}, conf_drop_threshold={args.conf_threshold}")

    results = []

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", f"sample_{idx}")
        print(f"\n===== Sample {idx}/{len(samples)} | id={sample_id} =====", flush=True)

        claim      = sample["claim"]
        evidence   = sample["evidence"]
        gold_label = get_gold_label(sample)

        # ── Direct prediction (conf 포함) ─────────────────────
        direct = direct_prediction_with_conf(claim, evidence)
        initial_answer = direct["answer"]
        initial_conf   = direct["label_conf"]

        conf_str = f"{initial_conf:.3f}" if initial_conf is not None else "N/A"

        print(f"  [Gold]   {gold_label}", flush=True)
        print(
            f"  [Direct] {direct['label']} (conf={conf_str})",
            flush=True,
        )

        row = {
            "id":         sample_id,
            "dataset":    dataset_name,
            "claim":      claim,
            "evidence":   evidence,
            "gold_label": gold_label,
            "source":     sample.get("source", ""),
            "need_review":sample.get("need_review", None),
            "comment":    sample.get("comment", ""),
            "direct":     direct,
        }

        # ── 기존 방식들 ───────────────────────────────────────
        row["basic_self_refine"] = basic_self_refinement(claim, evidence, initial_answer)
        print(f"  [Basic]          {row['basic_self_refine']['refined_label']}", flush=True)

        row["evidence_aware_self_refine"] = evidence_aware_self_refinement(claim, evidence, initial_answer)
        print(f"  [Evidence-aware] {row['evidence_aware_self_refine']['refined_label']}", flush=True)

        row["vitaminc_style_self_refine"] = vitaminc_style_self_refinement(claim, evidence, initial_answer)
        print(f"  [VitaminC]       {row['vitaminc_style_self_refine']['refined_label']}", flush=True)

        # ── WordR 방식 ────────────────────────────────────────
        row["wordr_self_refine"] = wordr_self_refinement(
            claim                     = claim,
            evidence                  = evidence,
            initial_answer            = initial_answer,
            initial_conf              = initial_conf,
            top_k                     = args.top_k,
            confidence_drop_threshold = args.conf_threshold,
            verbose                   = True,
        )
        row["wordr_self_refine"]["initial_label"] = normalize_label(
            row["wordr_self_refine"].get("initial_label")
        )

        row["wordr_self_refine"]["refined_label"] = normalize_label(
            row["wordr_self_refine"].get("refined_label")
        )        
        # initial_conf를 direct에서 받아 override (더 정확한 값)
        row["wordr_self_refine"]["initial_conf"] = initial_conf

        print(
            f"  [WordR]          {row['wordr_self_refine']['refined_label']} "
            f"| verified={row['wordr_self_refine']['verified_spans']}",
            flush=True,
        )

        results.append(row)

    # ── 정확도 출력 ───────────────────────────────────────────
    print("\n===== Accuracy =====")
    methods = [
        "direct",
        "basic_self_refine",
        "evidence_aware_self_refine",
        "vitaminc_style_self_refine",
        "wordr_self_refine",
    ]
    for method in methods:
        acc       = evaluate(results, method)
        pred_dist = label_counter(results, method)
        print(f"  {method:35s}: {acc:.3f} | {dict(pred_dist)}")

    # ── WordR 분석 통계 ───────────────────────────────────────
    print("\n===== WordR Analysis =====")

    total_cands   = sum(r["wordr_self_refine"]["total_candidates"] for r in results)
    total_verified= sum(r["wordr_self_refine"]["num_verified"] for r in results)
    zero_verified = sum(1 for r in results if r["wordr_self_refine"]["num_verified"] == 0)
    label_changed = sum(1 for r in results if r["wordr_self_refine"]["label_changed_after_refine"])

    # rationale_type 분포
    type_counter: Counter = Counter()
    for row in results:
        for rat in row["wordr_self_refine"]["verified_rationales"]:
            type_counter[rat["rationale_type"]] += 1

    # flip vs conf_drop 기반 정정 성공률
    flip_correct = 0
    conf_correct = 0
    flip_total   = 0
    conf_total   = 0

    for row in results:
        gold    = row["gold_label"]
        wordr   = row["wordr_self_refine"]
        refined = wordr["refined_label"]
        rats    = wordr["verified_rationales"]

        has_flip = any(r["rationale_type"] in ("flip_to_refutes", "flip_to_nei") for r in rats)
        has_conf = any(r["rationale_type"] == "confidence_drop" for r in rats) and not has_flip

        if has_flip:
            flip_total += 1
            if refined == gold:
                flip_correct += 1
        elif has_conf:
            conf_total += 1
            if refined == gold:
                conf_correct += 1

    print(f"  Extraction method          : {'spacy' if results and results[0]['wordr_self_refine']['extraction_method'] == 'spacy' else 'regex'}")
    print(f"  Total candidates tested    : {total_cands}")
    print(f"  Total verified rationales  : {total_verified}")
    print(f"  Avg verified / sample      : {total_verified / len(results):.2f}")
    print(f"  Samples with 0 verified    : {zero_verified}")
    print(f"  Label changed after refine : {label_changed}")
    print(f"  Rationale type dist        : {dict(type_counter)}")
    print(
        f"  Accuracy (flip-based)      : "
        f"{flip_correct}/{flip_total} = {flip_correct/flip_total:.3f}" if flip_total else "  Accuracy (flip-based): N/A"
    )
    print(
        f"  Accuracy (conf-drop-based) : "
        f"{conf_correct}/{conf_total} = {conf_correct/conf_total:.3f}" if conf_total else "  Accuracy (conf-drop-based): N/A"
    )

    # ── 저장 ─────────────────────────────────────────────────
    model_name = get_model_short_name()
    out_path = Path(args.output) if args.output else Path("results") / dataset_name / f"{model_name}_{dataset_name}_wordr.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} already exists. Use --overwrite to replace it.")

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()