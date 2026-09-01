import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


# LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
LABELS = ["SUPPORT", "REFUTE", "NOT ENOUGH INFO"]

def normalize_label(label):
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

    if x in {
        "NOT ENOUGH INFO", "NEI", "NOT ENOUGH INFORMATION",
        "INSUFFICIENT INFO"
    }:
        return "NOT ENOUGH INFO"

    return "PARSE_ERROR"

# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# Prediction getter
# ============================================================

def get_gold(row):
    """Return normalized gold label."""
    return normalize_label(row.get("gold_label"))


def get_pred(row, method_name):
    """
    method_name:
        direct
        basic_self_refine
        evidence_aware_self_refine
        vitaminc_style_self_refine
        wordr_self_refine
    
    Always return normalized 3-class label.
    """
    if method_name == "direct":
        return normalize_label(row.get("direct", {}).get("label"))

    return normalize_label(row.get(method_name, {}).get("refined_label"))


def get_available_methods(results):
    """
    결과 파일에 실제로 존재하는 method만 자동 탐지.
    baseline을 주석 처리하고 direct + wordr만 돌린 경우에도 평가 가능.
    """
    if not results:
        return []

    first = results[0]

    candidates = [
        "direct",
        "basic_self_refine",
        "evidence_aware_self_refine",
        "vitaminc_style_self_refine",
        "wordr_self_refine",
    ]

    methods = []
    for m in candidates:
        if m == "direct":
            if "direct" in first:
                methods.append(m)
        else:
            if m in first:
                methods.append(m)

    return methods


# ============================================================
# Basic classification metrics
# ============================================================

def accuracy(results, method_name):
    correct = 0
    total = 0

    for row in results:
        gold = get_gold(row)
        pred = get_pred(row, method_name)

        if pred == gold:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def classification_report_simple(results, method_name):
    report = {}

    for label in LABELS:
        tp = fp = fn = 0

        for row in results:
            gold = get_gold(row)
            pred = get_pred(row, method_name)

            if pred == label and gold == label:
                tp += 1
            elif pred == label and gold != label:
                fp += 1
            elif pred != label and gold == label:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        support = sum(1 for row in results if get_gold(row) == label)

        report[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    macro_f1 = sum(report[label]["f1"] for label in LABELS) / len(LABELS)

    return report, macro_f1


def confusion_matrix(results, method_name):
    matrix = {
        gold: {pred: 0 for pred in LABELS + ["PARSE_ERROR", "OTHER"]}
        for gold in LABELS
    }

    for row in results:
        gold = get_gold(row)
        pred = get_pred(row, method_name)

        if gold not in matrix:
            continue

        if pred not in matrix[gold]:
            pred = "OTHER"

        matrix[gold][pred] += 1

    return matrix


def prediction_distribution(results, method_name):
    preds = [get_pred(row, method_name) for row in results]
    return Counter(preds)


def parse_error_rate(results, method_name):
    total = len(results)
    if total == 0:
        return 0.0

    parse_errors = sum(
        1 for row in results
        if get_pred(row, method_name) == "PARSE_ERROR"
    )

    return parse_errors / total


# ============================================================
# Refinement metrics
# ============================================================

def compute_refinement_stats(results, method_name):
    """
    direct prediction 대비 refinement method의 변화 분석.

    C_to_C: direct 맞음  -> refined 맞음
    C_to_W: direct 맞음  -> refined 틀림
    W_to_C: direct 틀림  -> refined 맞음
    W_to_W: direct 틀림  -> refined 틀림
    """
    stats = {
        "C_to_C": 0,
        "C_to_W": 0,
        "W_to_C": 0,
        "W_to_W": 0,
        "label_changed": 0,
        "label_change_correct": 0,
    }

    for row in results:
        gold = get_gold(row)

        direct_pred = get_pred(row, "direct")
        refined_pred = get_pred(row, method_name)

        direct_correct = direct_pred == gold
        refined_correct = refined_pred == gold

        if direct_correct and refined_correct:
            stats["C_to_C"] += 1
        elif direct_correct and not refined_correct:
            stats["C_to_W"] += 1
        elif not direct_correct and refined_correct:
            stats["W_to_C"] += 1
        else:
            stats["W_to_W"] += 1

        if direct_pred != refined_pred:
            stats["label_changed"] += 1
            if refined_correct:
                stats["label_change_correct"] += 1

    initial_wrong = stats["W_to_C"] + stats["W_to_W"]
    initial_correct = stats["C_to_C"] + stats["C_to_W"]

    stats["correction_rate"] = (
        stats["W_to_C"] / initial_wrong if initial_wrong > 0 else 0.0
    )

    stats["degradation_rate"] = (
        stats["C_to_W"] / initial_correct if initial_correct > 0 else 0.0
    )

    stats["net_improvement"] = stats["W_to_C"] - stats["C_to_W"]

    stats["net_improvement_rate"] = (
        stats["net_improvement"] / len(results) if results else 0.0
    )

    stats["label_change_accuracy"] = (
        stats["label_change_correct"] / stats["label_changed"]
        if stats["label_changed"] > 0 else 0.0
    )

    return stats


# ============================================================
# WordR-specific metrics
# ============================================================

def compute_wordr_stats(results):
    total_samples = len(results)

    total_candidates = 0
    total_verified = 0
    zero_verified = 0
    label_changed_after_refine = 0

    type_counter = Counter()

    with_verified = []
    without_verified = []

    by_type_rows = defaultdict(list)

    for row in results:
        if "wordr_self_refine" not in row:
            continue

        wordr = row["wordr_self_refine"]

        total_candidates += wordr.get("total_candidates", 0)
        total_verified += wordr.get("num_verified", 0)

        if wordr.get("num_verified", 0) == 0:
            zero_verified += 1
            without_verified.append(row)
        else:
            with_verified.append(row)

        if wordr.get("label_changed_after_refine", False):
            label_changed_after_refine += 1

        for rat in wordr.get("verified_rationales", []):
            rtype = rat.get("rationale_type", "UNKNOWN")
            type_counter[rtype] += 1
            by_type_rows[rtype].append(row)

    verified_rate = (
        total_verified / total_candidates if total_candidates > 0 else 0.0
    )

    avg_verified_per_sample = (
        total_verified / total_samples if total_samples > 0 else 0.0
    )

    zero_rationale_rate = (
        zero_verified / total_samples if total_samples > 0 else 0.0
    )

    stats = {
        "total_candidates": total_candidates,
        "total_verified": total_verified,
        "verified_rationale_rate": verified_rate,
        "avg_verified_per_sample": avg_verified_per_sample,
        "zero_verified_samples": zero_verified,
        "zero_rationale_rate": zero_rationale_rate,
        "label_changed_after_refine": label_changed_after_refine,
        "rationale_type_distribution": dict(type_counter),
    }

    # verified rationale 유무별 refinement stats
    if with_verified:
        stats["with_verified_rationale"] = compute_refinement_stats(
            with_verified, "wordr_self_refine"
        )
    else:
        stats["with_verified_rationale"] = None

    if without_verified:
        stats["without_verified_rationale"] = compute_refinement_stats(
            without_verified, "wordr_self_refine"
        )
    else:
        stats["without_verified_rationale"] = None

    # rationale type별 correction/degradation 분석
    type_stats = {}
    for rtype, rows in by_type_rows.items():
        # 같은 row가 여러 번 들어갈 수 있으므로 id 기준 dedup
        dedup = {}
        for r in rows:
            dedup[r["id"]] = r
        unique_rows = list(dedup.values())

        type_stats[rtype] = compute_refinement_stats(
            unique_rows, "wordr_self_refine"
        )

    stats["rationale_type_refinement_stats"] = type_stats

    return stats


# ============================================================
# Error analysis buckets
# ============================================================

def assign_transition_bucket(row, method_name):
    gold = get_gold(row)
    direct_pred = get_pred(row, "direct")
    refined_pred = get_pred(row, method_name)

    direct_correct = direct_pred == gold
    refined_correct = refined_pred == gold

    if direct_correct and refined_correct:
        return "C_to_C"
    elif direct_correct and not refined_correct:
        return "C_to_W"
    elif not direct_correct and refined_correct:
        return "W_to_C"
    else:
        return "W_to_W"


def make_error_analysis_item(row, method_name):
    item = {
        "id": row.get("id"),
        "claim": row.get("claim"),
        "evidence": row.get("evidence"),
        "gold_label": get_gold(row),
        "direct_label": get_pred(row, "direct"),
        "method_label": get_pred(row, method_name),
        "transition": assign_transition_bucket(row, method_name),
        "source": row.get("source", ""),
        "comment": row.get("comment", ""),
        "direct_answer": row.get("direct", {}).get("answer", ""),
    }

    if method_name != "direct":
        item["refined_answer"] = row.get(method_name, {}).get("refined_answer", "")
        item["feedback"] = row.get(method_name, {}).get("feedback", "")

    if method_name == "wordr_self_refine":
        wordr = row.get("wordr_self_refine", {})
        item["verified_spans"] = wordr.get("verified_spans", [])
        item["verified_rationales"] = wordr.get("verified_rationales", [])
        item["num_verified"] = wordr.get("num_verified", 0)
        item["claim_candidates"] = wordr.get("claim_candidates", [])
        item["evidence_candidates"] = wordr.get("evidence_candidates", [])

    return item


def save_error_analysis_files(results, method_name, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    buckets = {
        "C_to_C": [],
        "C_to_W": [],
        "W_to_C": [],
        "W_to_W": [],
    }

    for row in results:
        bucket = assign_transition_bucket(row, method_name)
        item = make_error_analysis_item(row, method_name)
        buckets[bucket].append(item)

    for bucket, items in buckets.items():
        path = out_dir / f"{method_name}_{bucket}.jsonl"
        write_jsonl(path, items)
        print(f"Saved {len(items):4d} samples to {path}")


# ============================================================
# Printing
# ============================================================

def print_classification_summary(results, methods):
    print("\n===== Classification Metrics =====")

    for method in methods:
        acc = accuracy(results, method)
        report, macro_f1 = classification_report_simple(results, method)
        pred_dist = prediction_distribution(results, method)
        perr = parse_error_rate(results, method)

        print(f"\n[{method}]")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Macro-F1       : {macro_f1:.4f}")
        print(f"  Parse error    : {perr:.4f}")
        print(f"  Pred dist      : {dict(pred_dist)}")

        print("  Per-class:")
        for label in LABELS:
            r = report[label]
            print(
                f"    {label:16s} "
                f"P={r['precision']:.4f} "
                f"R={r['recall']:.4f} "
                f"F1={r['f1']:.4f} "
                f"N={r['support']}"
            )


def print_confusion_matrices(results, methods):
    print("\n===== Confusion Matrix =====")

    for method in methods:
        matrix = confusion_matrix(results, method)

        print(f"\n[{method}]")
        header = ["Gold \\ Pred"] + LABELS + ["PARSE_ERROR", "OTHER"]
        print("  " + " | ".join(f"{h:16s}" for h in header))

        for gold in LABELS:
            row_values = [gold] + [
                str(matrix[gold][pred]) for pred in LABELS + ["PARSE_ERROR", "OTHER"]
            ]
            print("  " + " | ".join(f"{v:16s}" for v in row_values))


def print_refinement_summary(results, methods):
    print("\n===== Refinement Metrics vs Direct =====")

    for method in methods:
        if method == "direct":
            continue

        stats = compute_refinement_stats(results, method)

        print(f"\n[{method}]")
        print(f"  C→C                  : {stats['C_to_C']}")
        print(f"  C→W                  : {stats['C_to_W']}")
        print(f"  W→C                  : {stats['W_to_C']}")
        print(f"  W→W                  : {stats['W_to_W']}")
        print(f"  Correction rate      : {stats['correction_rate']:.4f}")
        print(f"  Degradation rate     : {stats['degradation_rate']:.4f}")
        print(f"  Net improvement      : {stats['net_improvement']}")
        print(f"  Net improvement rate : {stats['net_improvement_rate']:.4f}")
        print(f"  Label changed        : {stats['label_changed']}")
        print(f"  Label change acc     : {stats['label_change_accuracy']:.4f}")


def print_wordr_summary(results):
    if not results or "wordr_self_refine" not in results[0]:
        return

    stats = compute_wordr_stats(results)

    print("\n===== WordR Rationale Metrics =====")
    print(f"  Total candidates             : {stats['total_candidates']}")
    print(f"  Total verified rationales    : {stats['total_verified']}")
    print(f"  Verified rationale rate      : {stats['verified_rationale_rate']:.4f}")
    print(f"  Avg verified / sample        : {stats['avg_verified_per_sample']:.4f}")
    print(f"  Zero-rationale samples       : {stats['zero_verified_samples']}")
    print(f"  Zero-rationale rate          : {stats['zero_rationale_rate']:.4f}")
    print(f"  Label changed after refine   : {stats['label_changed_after_refine']}")
    print(f"  Rationale type dist          : {stats['rationale_type_distribution']}")

    print("\n  [With verified rationale]")
    if stats["with_verified_rationale"] is not None:
        s = stats["with_verified_rationale"]
        print(f"    Correction rate  : {s['correction_rate']:.4f}")
        print(f"    Degradation rate : {s['degradation_rate']:.4f}")
        print(f"    Net improvement  : {s['net_improvement']}")
    else:
        print("    N/A")

    print("\n  [Without verified rationale]")
    if stats["without_verified_rationale"] is not None:
        s = stats["without_verified_rationale"]
        print(f"    Correction rate  : {s['correction_rate']:.4f}")
        print(f"    Degradation rate : {s['degradation_rate']:.4f}")
        print(f"    Net improvement  : {s['net_improvement']}")
    else:
        print("    N/A")

    print("\n  [By rationale type]")
    for rtype, s in stats["rationale_type_refinement_stats"].items():
        print(f"    {rtype}")
        print(f"      Correction rate  : {s['correction_rate']:.4f}")
        print(f"      Degradation rate : {s['degradation_rate']:.4f}")
        print(f"      Net improvement  : {s['net_improvement']}")


# ============================================================
# Save summary JSON
# ============================================================

def build_summary(results, methods):
    summary = {
        "num_samples": len(results),
        "gold_distribution": dict(Counter(get_gold(row) for row in results)),
        "methods": {},
    }

    for method in methods:
        acc = accuracy(results, method)
        report, macro_f1 = classification_report_simple(results, method)

        summary["methods"][method] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "parse_error_rate": parse_error_rate(results, method),
            "prediction_distribution": dict(prediction_distribution(results, method)),
            "per_class": report,
            "confusion_matrix": confusion_matrix(results, method),
        }

        if method != "direct":
            summary["methods"][method]["refinement"] = compute_refinement_stats(
                results, method
            )

    if "wordr_self_refine" in methods:
        summary["wordr_rationale_metrics"] = compute_wordr_stats(results)

    return summary


def save_summary_json(summary, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nSaved summary to {path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to result jsonl from run_experiment_wordr.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--save-error-analysis",
        action="store_true",
        help="Save C_to_C, C_to_W, W_to_C, W_to_W jsonl files",
    )
    parser.add_argument(
        "--method-for-error-analysis",
        type=str,
        default="wordr_self_refine",
        help="Method to use for error analysis buckets",
    )

    args = parser.parse_args()

    results = read_jsonl(args.input)
    methods = get_available_methods(results)

    print(f"Loaded {len(results)} results from {args.input}")
    print(f"Available methods: {methods}")
    print("Gold distribution:", dict(Counter(get_gold(row) for row in results)))

    print_classification_summary(results, methods)
    print_confusion_matrices(results, methods)
    print_refinement_summary(results, methods)
    print_wordr_summary(results)

    summary = build_summary(results, methods)

    input_stem = Path(args.input).stem
    summary_path = Path(args.out_dir) / f"{input_stem}_evaluation_summary.json"
    save_summary_json(summary, summary_path)

    if args.save_error_analysis:
        method = args.method_for_error_analysis

        if method not in methods:
            raise ValueError(
                f"Method '{method}' not found in result file. "
                f"Available methods: {methods}"
            )

        error_dir = Path(args.out_dir) / f"{input_stem}_error_analysis"
        save_error_analysis_files(results, method, error_dir)


if __name__ == "__main__":
    main()