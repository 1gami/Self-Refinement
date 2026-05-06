import json
import re
from collections import Counter, defaultdict

INPUT_PATH = "results/scifact/ablation_qwen_scifact_word_cand.jsonl"

METHODS = [
    "direct",
    "basic_self_refine",
    "evidence_aware_self_refine",
    "vitaminc_style_self_refine",
    "wordcand_self_refine",
]

BASELINE = "direct"

LABELS = ["SUPPORT", "REFUTE", "NOT ENOUGH INFO"]


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

    m = re.search(
        r"final\s+label\s*:\s*(NOT ENOUGH INFO|SUPPORTS?|REFUTES?|NEI)",
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
            "pred_label",
            "answer_label",
            "initial_label",
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
        ]:
            if key in value:
                lab = extract_label_from_text(value.get(key))
                if lab != "PARSE_ERROR":
                    return lab

    return "PARSE_ERROR"


def compute_metrics(y_true, y_pred):
    total = len(y_true)
    correct = sum(g == p for g, p in zip(y_true, y_pred))
    acc = correct / total if total else 0.0

    per_label = {}

    f1s = []
    recalls = []
    precisions = []

    for label in LABELS:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == label and p != label)

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        support = sum(1 for g in y_true if g == label)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions)
    macro_recall = sum(recalls) / len(recalls)
    macro_f1 = sum(f1s) / len(f1s)

    return {
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_label": per_label,
        "pred_dist": Counter(y_pred),
    }


def main():
    rows = []

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    y_true = [normalize_label(ex.get("gold_label")) for ex in rows]

    preds = {}
    metrics = {}

    for method in METHODS:
        y_pred = [extract_pred_label(ex.get(method)) for ex in rows]
        preds[method] = y_pred
        metrics[method] = compute_metrics(y_true, y_pred)

    print(f"Loaded {len(rows)} examples from {INPUT_PATH}")
    print()

    print("===== Overall Performance =====")
    print(
        f"{'method':<32} {'acc':>8} {'macro_f1':>10} {'macro_p':>10} {'macro_r':>10} {'correct':>10}"
    )

    for method in METHODS:
        m = metrics[method]
        print(
            f"{method:<32} "
            f"{m['accuracy']:>8.3f} "
            f"{m['macro_f1']:>10.3f} "
            f"{m['macro_precision']:>10.3f} "
            f"{m['macro_recall']:>10.3f} "
            f"{m['correct']:>5}/{m['total']:<4}"
        )

    print()
    print("===== Delta vs Direct =====")
    base_pred = preds[BASELINE]
    base_acc = metrics[BASELINE]["accuracy"]
    base_f1 = metrics[BASELINE]["macro_f1"]

    print(
        f"{'method':<32} {'acc_delta':>10} {'f1_delta':>10} "
        f"{'fixed':>8} {'broken':>8} {'same_correct':>14} {'same_wrong':>12}"
    )

    for method in METHODS:
        if method == BASELINE:
            continue

        y_pred = preds[method]

        fixed = 0
        broken = 0
        same_correct = 0
        same_wrong = 0

        for g, b, p in zip(y_true, base_pred, y_pred):
            b_ok = g == b
            p_ok = g == p

            if not b_ok and p_ok:
                fixed += 1
            elif b_ok and not p_ok:
                broken += 1
            elif b_ok and p_ok:
                same_correct += 1
            else:
                same_wrong += 1

        acc_delta = metrics[method]["accuracy"] - base_acc
        f1_delta = metrics[method]["macro_f1"] - base_f1

        print(
            f"{method:<32} "
            f"{acc_delta:>+10.3f} "
            f"{f1_delta:>+10.3f} "
            f"{fixed:>8} "
            f"{broken:>8} "
            f"{same_correct:>14} "
            f"{same_wrong:>12}"
        )

    print()
    print("===== Per-label F1 =====")
    print(f"{'method':<32} {'SUPPORT':>10} {'REFUTE':>10} {'NEI':>10}")

    for method in METHODS:
        per = metrics[method]["per_label"]
        print(
            f"{method:<32} "
            f"{per['SUPPORT']['f1']:>10.3f} "
            f"{per['REFUTE']['f1']:>10.3f} "
            f"{per['NOT ENOUGH INFO']['f1']:>10.3f}"
        )

    print()
    print("===== Prediction Distribution =====")
    for method in METHODS:
        print(method, dict(metrics[method]["pred_dist"]))

    print()
    print("===== Main Error Types by Method =====")

    for method in METHODS:
        err_counter = Counter()

        for g, p in zip(y_true, preds[method]):
            if g != p:
                err_counter[(g, p)] += 1

        print()
        print(method)
        for (g, p), c in err_counter.most_common(10):
            print(f"  gold={g:<16} pred={p:<16} count={c}")


if __name__ == "__main__":
    main()