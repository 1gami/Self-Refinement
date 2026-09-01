"""Summarize WordR grid-search result files."""

import argparse
import csv
import glob
import json
import re
from pathlib import Path

METHODS = [
    "direct",
    "basic_self_refine",
    "evidence_aware_self_refine",
    "vitaminc_style_self_refine",
    "wordr_self_refine",
]


def normalize_label(value):
    if value is None:
        return "PARSE_ERROR"
    label = str(value).strip().upper().replace("_", " ")
    mapping = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "REFUTED": "REFUTE",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        "NEI": "NOT ENOUGH INFO",
    }
    return mapping.get(label, "PARSE_ERROR")


def extract_label_from_text(text):
    if text is None:
        return "PARSE_ERROR"
    match = re.search(
        r"label\s*:\s*(NOT ENOUGH INFO|SUPPORTS?|REFUTES?|NEI)",
        str(text),
        flags=re.IGNORECASE,
    )
    return normalize_label(match.group(1) if match else text)


def extract_pred_label(value):
    if value is None:
        return "PARSE_ERROR"
    if isinstance(value, str):
        return extract_label_from_text(value)
    if isinstance(value, dict):
        for key in (
            "final_label", "refined_label", "label", "prediction", "pred",
            "pred_label", "answer_label", "initial_label", "final_prediction",
            "refined_prediction", "verdict", "final_verdict", "refined_verdict",
        ):
            if key in value:
                label = normalize_label(value.get(key))
                if label != "PARSE_ERROR":
                    return label
        for key in (
            "answer", "final_answer", "refined_answer", "refinement", "output",
            "response", "initial_answer", "final_output", "refined_output", "text",
        ):
            if key in value:
                label = extract_label_from_text(value.get(key))
                if label != "PARSE_ERROR":
                    return label
    return "PARSE_ERROR"


def parse_params(path):
    match = re.search(r"topk(\d+)_thr(\d+)p(\d+)", str(path))
    if not match:
        return None, None
    return int(match.group(1)), float(f"{match.group(2)}.{match.group(3)}")


def accuracy(rows, method):
    total = len(rows)
    correct = sum(
        normalize_label(row.get("gold_label")) == extract_pred_label(row.get(method))
        for row in rows
    )
    return correct / total if total else 0.0, correct, total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        default="results/grid_wordr",
        help="Grid-search root directory. If --dataset is set, its subdirectory is used.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset name used in grid output filenames")
    parser.add_argument("--output", default=None, help="Summary CSV path")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if args.dataset:
        result_dir = result_dir / args.dataset
        pattern = f"{args.dataset}_topk*_thr*.jsonl"
    else:
        pattern = "*_topk*_thr*.jsonl"

    files = sorted(glob.glob(str(result_dir / pattern)))
    if not files:
        raise FileNotFoundError(f"No grid result files matching {result_dir / pattern}")

    summary_rows = []
    for path in files:
        top_k, threshold = parse_params(path)
        if top_k is None:
            continue
        with open(path, "r", encoding="utf-8") as file:
            rows = [json.loads(line) for line in file if line.strip()]

        result = {"path": path, "top_k": top_k, "threshold": threshold}
        for method in METHODS:
            acc, correct, total = accuracy(rows, method)
            result[method] = acc
            result[f"{method}_correct"] = correct
            result["total"] = total
        result["wordr_minus_direct"] = result["wordr_self_refine"] - result["direct"]
        summary_rows.append(result)

    summary_rows.sort(key=lambda row: (row["top_k"], row["threshold"]))
    print("===== Ablation Summary =====")
    print(f"{'top_k':>5} {'thr':>6} {'direct':>8} {'basic':>8} {'evid':>8} {'vitc':>8} {'wordr':>8} {'w-d':>8}")
    for row in summary_rows:
        print(
            f"{row['top_k']:>5} {row['threshold']:>6.2f} {row['direct']:>8.3f} "
            f"{row['basic_self_refine']:>8.3f} {row['evidence_aware_self_refine']:>8.3f} "
            f"{row['vitaminc_style_self_refine']:>8.3f} {row['wordr_self_refine']:>8.3f} "
            f"{row['wordr_minus_direct']:>+8.3f}"
        )

    best = max(summary_rows, key=lambda row: row["wordr_self_refine"])
    print("\n===== Best WordR Setting =====")
    print(f"top_k={best['top_k']}, threshold={best['threshold']}")
    print(f"WordR accuracy: {best['wordr_self_refine']:.3f}")
    print(f"Direct accuracy: {best['direct']:.3f}")
    print(f"Delta: {best['wordr_minus_direct']:+.3f}")

    output_path = Path(args.output) if args.output else result_dir / "ablation_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "top_k", "threshold", "direct", "basic_self_refine", "evidence_aware_self_refine",
        "vitaminc_style_self_refine", "wordr_self_refine", "wordr_minus_direct", "total",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved CSV to {output_path}")


if __name__ == "__main__":
    main()
