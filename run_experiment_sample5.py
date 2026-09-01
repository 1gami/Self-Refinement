import json
import os
import re
from pathlib import Path

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


def parse_label(output):
    # Label: REFUTES 또는 Label: [REFUTES] 둘 다 처리
    match = re.search(r'Label:\s*\[?\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO|NEI)\s*\]?', output, re.IGNORECASE)
    if match:
        label = match.group(1).upper().strip()
        if label == "NEI":
            return "NOT ENOUGH INFO"
        return label
    return "PARSE_ERROR"


def direct_prediction(claim, evidence):
    answer = call_llm(direct_prompt(claim, evidence))
    return {
        "answer": answer,
        "label": parse_label(answer),
    }


def basic_self_refinement(claim, evidence, initial_answer):
    feedback = call_llm(basic_feedback_prompt(claim, evidence, initial_answer))
    refined_answer = call_llm(basic_refine_prompt(claim, evidence, initial_answer, feedback))
    return {
        "initial_answer": initial_answer,
        "feedback": feedback,
        "refined_answer": refined_answer,
        "initial_label": parse_label(initial_answer),
        "refined_label": parse_label(refined_answer),
    }


def evidence_aware_self_refinement(claim, evidence, initial_answer):
    feedback = call_llm(evidence_aware_feedback_prompt(claim, evidence, initial_answer))
    refined_answer = call_llm(evidence_aware_refine_prompt(claim, evidence, initial_answer, feedback))
    return {
        "initial_answer": initial_answer,
        "feedback": feedback,
        "refined_answer": refined_answer,
        "initial_label": parse_label(initial_answer),
        "refined_label": parse_label(refined_answer),
    }


def vitaminc_style_self_refinement(claim, evidence, initial_answer):
    feedback = call_llm(vitaminc_style_feedback_prompt(claim, evidence, initial_answer))
    refined_answer = call_llm(vitaminc_style_refine_prompt(claim, evidence, initial_answer, feedback))
    return {
        "initial_answer": initial_answer,
        "feedback": feedback,
        "refined_answer": refined_answer,
        "initial_label": parse_label(initial_answer),
        "refined_label": parse_label(refined_answer),
    }


def evaluate(results, method_name):
    correct = 0
    total = 0
    for row in results:
        gold = row["gold_label"]
        if method_name == "direct":
            pred = row["direct"]["label"]
        else:
            pred = row[method_name]["refined_label"]
        if pred == gold:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0


def get_model_short_name():
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    return Path(model_path).name.replace("/", "_")


def main():
    samples = [
        {
            "id": "ex1",
            "claim": "Aspirin reduces the risk of heart attack.",
            "evidence": "Aspirin can reduce platelet aggregation, which may lower the risk of heart attack in some patients.",
            "gold_label": "SUPPORTS",
        },
        {
            "id": "ex2",
            "claim": "The drug increased blood pressure in all patients.",
            "evidence": "The drug decreased systolic blood pressure in most patients during the trial.",
            "gold_label": "REFUTES",
        },
        {
            "id": "ex3",
            "claim": "Protein X causes liver cancer.",
            "evidence": "Protein X was observed at higher levels in patients with liver cancer, but the study did not establish causality.",
            "gold_label": "NOT ENOUGH INFO",
        },
        {
            "id": "ex4",
            "claim": "Treatment A improved survival after 12 months.",
            "evidence": "Treatment A improved survival after 6 months, but no significant difference was found after 12 months.",
            "gold_label": "REFUTES",
        },
        {
            "id": "ex5",
            "claim": "Gene A inhibits tumor growth.",
            "evidence": "Gene A promotes tumor growth in the studied mouse model.",
            "gold_label": "REFUTES",
        },
    ]

    results = []

    for sample in samples:
        print(f"\n===== Running sample {sample['id']} =====", flush=True)

        claim = sample["claim"]
        evidence = sample["evidence"]

        # direct prediction (initial_answer 한 번만 생성)
        direct = direct_prediction(claim, evidence)
        initial_answer = direct["answer"]

        print(f"  [Direct] {direct['label']}", flush=True)

        row = {
            "id": sample["id"],
            "claim": claim,
            "evidence": evidence,
            "gold_label": sample["gold_label"],
            "direct": direct,
        }

        row["basic_self_refine"] = basic_self_refinement(claim, evidence, initial_answer)
        print(f"  [Basic Self-Refine] {row['basic_self_refine']['refined_label']}", flush=True)

        row["evidence_aware_self_refine"] = evidence_aware_self_refinement(claim, evidence, initial_answer)
        print(f"  [Evidence-aware] {row['evidence_aware_self_refine']['refined_label']}", flush=True)

        row["vitaminc_style_self_refine"] = vitaminc_style_self_refinement(claim, evidence, initial_answer)
        print(f"  [VitaminC-style] {row['vitaminc_style_self_refine']['refined_label']}", flush=True)

        results.append(row)

    print("\n===== Accuracy =====")
    for method in ["direct", "basic_self_refine", "evidence_aware_self_refine", "vitaminc_style_self_refine"]:
        print(f"  {method}: {evaluate(results, method):.3f}")

    model_name = get_model_short_name()
    out_path = Path("results") / f"{model_name}_self_refine_results.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()