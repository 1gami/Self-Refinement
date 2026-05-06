import json
import re
import glob
from collections import Counter

RESULT_DIR = "results/grid_wordr/fever"

METHODS = [
    "direct",
    "basic_self_refine",
    "evidence_aware_self_refine",
    "vitaminc_style_self_refine",
    "wordr_self_refine",
]

def normalize_label(x):
    if x is None:
        return "PARSE_ERROR"

    x = str(x).strip().upper()

    mapping = {
        "SUPPORT": "SUPPORT",
        "SUPPORTS": "SUPPORT",
        "SUPPORTED": "SUPPORT",
        "REFUTE": "REFUTE",
        "REFUTES": "REFUTE",
        "REFUTED": "REFUTE",
        "NOT ENOUGH INFO": "NOT ENOUGH INFO",
        "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
        "NEI": "NOT ENOUGH INFO",
    }
    return mapping.get(x, "PARSE_ERROR")


def extract_label_from_text(text):
    if text is None:
        return "PARSE_ERROR"

    text = str(text).strip()

    m = re.search(
        r"label\s*:\s*(NOT ENOUGH INFO|SUPPORTS?|REFUTES?|NEI)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        return normalize_label(m.group(1))

    return normalize_label(text)


def extract_pred_label(value):
    if value is None:
        return "PARSE_ERROR"

    if isinstance(value, str):
        return extract_label_from_text(value)

    if isinstance(value, dict):
        for key in [
            "final_label",
            "refined_label",
            "label",
            "prediction",
            "pred",
            "pred_label",
            "answer_label",
            "initial_label",
            "final_prediction",
            "refined_prediction",
            "verdict",
            "final_verdict",
            "refined_verdict",
        ]:
            if key in value:
                lab = normalize_label(value.get(key))
                if lab != "PARSE_ERROR":
                    return lab

        for key in [
            "answer",
            "final_answer",
            "refined_answer",
            "refinement",
            "output",
            "response",
            "initial_answer",
            "final_output",
            "refined_output",
            "text",
        ]:
            if key in value:
                lab = extract_label_from_text(value.get(key))
                if lab != "PARSE_ERROR":
                    return lab

        lab = extract_label_from_text(json.dumps(value, ensure_ascii=False))
        if lab != "PARSE_ERROR":
            return lab

    return "PARSE_ERROR"


def parse_params(path):
    # fever_topk3_thr0p1.jsonl -> top_k=3, threshold=0.1
    m = re.search(r"topk(\d+)_thr(\d+)p(\d+)", path)
    if not m:
        return None, None

    top_k = int(m.group(1))
    threshold = float(f"{m.group(2)}.{m.group(3)}")
    return top_k, threshold


def accuracy(rows, method):
    correct = 0
    total = 0

    for ex in rows:
        gold = normalize_label(ex.get("gold_label"))
        pred = extract_pred_label(ex.get(method))

        if gold == pred:
            correct += 1
        total += 1

    return correct / total if total else 0.0, correct, total


def main():
    files = sorted(glob.glob(f"{RESULT_DIR}/fever_topk*_thr*.jsonl"))

    if not files:
        print(f"No files found in {RESULT_DIR}")
        return

    rows_out = []

    for path in files:
        top_k, threshold = parse_params(path)

        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]

        result = {
            "path": path,
            "top_k": top_k,
            "threshold": threshold,
        }

        for method in METHODS:
            acc, correct, total = accuracy(rows, method)
            result[method] = acc
            result[f"{method}_correct"] = correct
            result["total"] = total

        result["wordr_minus_direct"] = (
            result["wordr_self_refine"] - result["direct"]
        )

        rows_out.append(result)

    rows_out.sort(key=lambda x: (x["top_k"], x["threshold"]))

    print("===== Ablation Summary =====")
    print(
        f"{'top_k':>5} {'thr':>6} "
        f"{'direct':>8} {'basic':>8} {'evid':>8} {'vitc':>8} {'wordr':>8} {'w-d':>8}"
    )

    for r in rows_out:
        print(
            f"{r['top_k']:>5} {r['threshold']:>6.1f} "
            f"{r['direct']:>8.3f} "
            f"{r['basic_self_refine']:>8.3f} "
            f"{r['evidence_aware_self_refine']:>8.3f} "
            f"{r['vitaminc_style_self_refine']:>8.3f} "
            f"{r['wordr_self_refine']:>8.3f} "
            f"{r['wordr_minus_direct']:>+8.3f}"
        )

    best = max(rows_out, key=lambda x: x["wordr_self_refine"])

    print()
    print("===== Best WordR Setting =====")
    print(f"top_k={best['top_k']}, threshold={best['threshold']}")
    print(f"WordR accuracy: {best['wordr_self_refine']:.3f}")
    print(f"Direct accuracy: {best['direct']:.3f}")
    print(f"Delta: {best['wordr_minus_direct']:+.3f}")

    out_csv = f"{RESULT_DIR}/ablation_summary.csv"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("top_k,threshold,direct,basic_self_refine,evidence_aware_self_refine,vitaminc_style_self_refine,wordr_self_refine,wordr_minus_direct,total\n")
        for r in rows_out:
            f.write(
                f"{r['top_k']},{r['threshold']},"
                f"{r['direct']:.6f},"
                f"{r['basic_self_refine']:.6f},"
                f"{r['evidence_aware_self_refine']:.6f},"
                f"{r['vitaminc_style_self_refine']:.6f},"
                f"{r['wordr_self_refine']:.6f},"
                f"{r['wordr_minus_direct']:.6f},"
                f"{r['total']}\n"
            )

    print()
    print(f"Saved CSV to {out_csv}")


if __name__ == "__main__":
    main()