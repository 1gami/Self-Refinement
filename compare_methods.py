"""Compare fact-verification methods stored in one experiment JSONL file."""

import argparse
import json
import re
from collections import Counter

LABELS = ["SUPPORT", "REFUTE", "NOT ENOUGH INFO"]
DEFAULT_METHOD_ORDER = [
    "direct",
    "basic_self_refine",
    "evidence_aware_self_refine",
    "vitaminc_style_self_refine",
    "wordr_self_refine",
    "wordcand_self_refine",
]


def normalize_label(value):
    if value is None:
        return "PARSE_ERROR"

    label = str(value).strip().upper().replace("_", " ")
    mapping = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "ENTAILMENT": "SUPPORT",
        "TRUE": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "REFUTED": "REFUTE",
        "CONTRADICT": "REFUTE",
        "CONTRADICTS": "REFUTE",
        "CONTRADICTION": "REFUTE",
        "FALSE": "REFUTE",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        "NOT ENOUGH INFORMATION": "NOT ENOUGH INFO",
        "INSUFFICIENT INFO": "NOT ENOUGH INFO",
        "NEI": "NOT ENOUGH INFO",
    }
    return mapping.get(label, "PARSE_ERROR")


def extract_label_from_text(text):
    if text is None:
        return "PARSE_ERROR"

    text = str(text).strip()
    match = re.search(
        r"(?:final\s+)?label\s*:\s*(NOT ENOUGH INFO|SUPPORTS?|REFUTES?|NEI)",
        text,
        flags=re.IGNORECASE,
    )
    if match:
        return normalize_label(match.group(1))
    return normalize_label(text)


def extract_pred_label(value):
    if value is None:
        return "PARSE_ERROR"
    if isinstance(value, str):
        return extract_label_from_text(value)
    if isinstance(value, dict):
        for key in (
            "final_label",
            "refined_label",
            "label",
            "prediction",
            "pred_label",
            "answer_label",
            "initial_label",
        ):
            if key in value:
                label = normalize_label(value.get(key))
                if label != "PARSE_ERROR":
                    return label
        for key in (
            "answer",
            "final_answer",
            "refined_answer",
            "refinement",
            "output",
            "response",
            "initial_answer",
        ):
            if key in value:
                label = extract_label_from_text(value.get(key))
                if label != "PARSE_ERROR":
                    return label
    return "PARSE_ERROR"


def compute_metrics(y_true, y_pred):
    total = len(y_true)
    correct = sum(gold == pred for gold, pred in zip(y_true, y_pred))
    per_label = {}
    precisions, recalls, f1s = [], [], []

    for label in LABELS:
        tp = sum(g == label and p == label for g, p in zip(y_true, y_pred))
        fp = sum(g != label and p == label for g, p in zip(y_true, y_pred))
        fn = sum(g == label and p != label for g, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(g == label for g in y_true),
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "per_label": per_label,
        "pred_dist": Counter(y_pred),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Experiment result JSONL")
    parser.add_argument("--baseline", default="direct")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Methods to compare. By default, methods present in the file are used.",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    if not rows:
        raise ValueError(f"No rows found in {args.input}")

    if args.methods:
        methods = args.methods
    else:
        methods = [method for method in DEFAULT_METHOD_ORDER if method in rows[0]]

    if args.baseline not in methods:
        raise ValueError(f"Baseline '{args.baseline}' is not available. Methods: {methods}")

    y_true = [normalize_label(row.get("gold_label")) for row in rows]
    predictions = {
        method: [extract_pred_label(row.get(method)) for row in rows]
        for method in methods
    }
    metrics = {
        method: compute_metrics(y_true, predictions[method])
        for method in methods
    }

    print(f"Loaded {len(rows)} examples from {args.input}\n")
    print("===== Overall Performance =====")
    print(f"{'method':<32} {'acc':>8} {'macro_f1':>10} {'macro_p':>10} {'macro_r':>10} {'correct':>10}")
    for method in methods:
        score = metrics[method]
        print(
            f"{method:<32} {score['accuracy']:>8.3f} {score['macro_f1']:>10.3f} "
            f"{score['macro_precision']:>10.3f} {score['macro_recall']:>10.3f} "
            f"{score['correct']:>5}/{score['total']:<4}"
        )

    baseline_pred = predictions[args.baseline]
    baseline_acc = metrics[args.baseline]["accuracy"]
    baseline_f1 = metrics[args.baseline]["macro_f1"]

    print("\n===== Delta vs Baseline =====")
    print(f"{'method':<32} {'acc_delta':>10} {'f1_delta':>10} {'fixed':>8} {'broken':>8} {'same_correct':>14} {'same_wrong':>12}")
    for method in methods:
        if method == args.baseline:
            continue
        fixed = broken = same_correct = same_wrong = 0
        for gold, base_pred, pred in zip(y_true, baseline_pred, predictions[method]):
            base_ok = gold == base_pred
            pred_ok = gold == pred
            if not base_ok and pred_ok:
                fixed += 1
            elif base_ok and not pred_ok:
                broken += 1
            elif base_ok and pred_ok:
                same_correct += 1
            else:
                same_wrong += 1
        print(
            f"{method:<32} {metrics[method]['accuracy'] - baseline_acc:>+10.3f} "
            f"{metrics[method]['macro_f1'] - baseline_f1:>+10.3f} {fixed:>8} {broken:>8} "
            f"{same_correct:>14} {same_wrong:>12}"
        )

    print("\n===== Per-label F1 =====")
    print(f"{'method':<32} {'SUPPORT':>10} {'REFUTE':>10} {'NEI':>10}")
    for method in methods:
        per_label = metrics[method]["per_label"]
        print(
            f"{method:<32} {per_label['SUPPORT']['f1']:>10.3f} "
            f"{per_label['REFUTE']['f1']:>10.3f} {per_label['NOT ENOUGH INFO']['f1']:>10.3f}"
        )

    print("\n===== Prediction Distribution =====")
    for method in methods:
        print(method, dict(metrics[method]["pred_dist"]))


if __name__ == "__main__":
    main()
