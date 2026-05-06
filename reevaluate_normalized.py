import argparse
import json
from collections import Counter, defaultdict


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
        "NOT ENOUGH INFO",
        "NEI",
        "NOT ENOUGH INFORMATION",
        "INSUFFICIENT INFO"
    }:
        return "NOT ENOUGH INFO"

    return "PARSE_ERROR"


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_pred(row, method):
    if method == "direct":
        return row["direct"].get("label")
    return row[method].get("refined_label")


def evaluate(rows, method):
    correct = 0
    total = 0
    pred_counter = Counter()
    gold_counter = Counter()
    confusion = defaultdict(Counter)

    for row in rows:
        gold = normalize_label(row.get("gold_label"))
        pred = normalize_label(get_pred(row, method))

        gold_counter[gold] += 1
        pred_counter[pred] += 1
        confusion[gold][pred] += 1

        if pred == gold:
            correct += 1
        total += 1

    acc = correct / total if total else 0.0
    return acc, pred_counter, gold_counter, confusion


def print_confusion(confusion):
    labels = ["SUPPORT", "REFUTE", "NOT ENOUGH INFO", "PARSE_ERROR"]

    print("\nConfusion matrix")
    print("gold \\ pred".ljust(20), end="")
    for p in labels:
        print(p[:12].rjust(14), end="")
    print()

    for g in labels:
        print(g.ljust(20), end="")
        for p in labels:
            print(str(confusion[g][p]).rjust(14), end="")
        print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--save-normalized", default=None)
    args = parser.parse_args()

    rows = read_jsonl(args.input)

    methods = [
        "direct",
        "basic_self_refine",
        "evidence_aware_self_refine",
        "vitaminc_style_self_refine",
        "wordcand_self_refine",
    ]

    print(f"Loaded {len(rows)} rows from {args.input}")

    print("\n===== Re-evaluated Accuracy with Normalized Labels =====")
    for method in methods:
        acc, pred_counter, gold_counter, confusion = evaluate(rows, method)
        print(f"{method:35s}: {acc:.3f} | {dict(pred_counter)}")

    # WordCand detailed confusion
    _, _, _, wordcand_confusion = evaluate(rows, "wordcand_self_refine")
    print_confusion(wordcand_confusion)

    # WordCand rationale analysis도 normalize해서 다시 계산
    print("\n===== Re-evaluated WordCand Analysis =====")

    flip_correct = 0
    conf_correct = 0
    flip_total = 0
    conf_total = 0

    label_changed = 0
    total_verified = 0
    total_candidates = 0
    zero_verified = 0
    zero_candidates = 0

    rationale_type_counter = Counter()
    candidate_type_counter = Counter()

    for row in rows:
        gold = normalize_label(row.get("gold_label"))
        wordcand = row.get("wordcand_self_refine", {})
        refined = normalize_label(wordcand.get("refined_label"))
        initial = normalize_label(wordcand.get("initial_label"))

        if initial != refined:
            label_changed += 1

        total_candidates += wordcand.get("total_candidates", 0)
        total_verified += wordcand.get("num_verified", 0)

        if wordcand.get("total_candidates", 0) == 0:
            zero_candidates += 1
        if wordcand.get("num_verified", 0) == 0:
            zero_verified += 1

        for cand in wordcand.get("candidate_rationales", []):
            candidate_type_counter[cand.get("difference_type", "unknown")] += 1

        rats = wordcand.get("verified_rationales", [])
        for rat in rats:
            rationale_type_counter[rat.get("rationale_type", "unknown")] += 1

        has_flip = any(
            r.get("rationale_type") in ("flip_to_refutes", "flip_to_nei")
            for r in rats
        )
        has_conf = any(
            r.get("rationale_type") == "confidence_drop"
            for r in rats
        ) and not has_flip

        if has_flip:
            flip_total += 1
            if refined == gold:
                flip_correct += 1
        elif has_conf:
            conf_total += 1
            if refined == gold:
                conf_correct += 1

    n = len(rows)
    print(f"Total candidates tested    : {total_candidates}")
    print(f"Total verified rationales  : {total_verified}")
    print(f"Avg candidates / sample    : {total_candidates / n:.2f}" if n else "N/A")
    print(f"Avg verified / sample      : {total_verified / n:.2f}" if n else "N/A")
    print(f"Samples with 0 candidates  : {zero_candidates}")
    print(f"Samples with 0 verified    : {zero_verified}")
    print(f"Label changed after refine : {label_changed}")
    print(f"Candidate type dist        : {dict(candidate_type_counter)}")
    print(f"Rationale type dist        : {dict(rationale_type_counter)}")

    if flip_total:
        print(f"Accuracy flip-based        : {flip_correct}/{flip_total} = {flip_correct / flip_total:.3f}")
    else:
        print("Accuracy flip-based        : N/A")

    if conf_total:
        print(f"Accuracy conf-drop-based   : {conf_correct}/{conf_total} = {conf_correct / conf_total:.3f}")
    else:
        print("Accuracy conf-drop-based   : N/A")

    # 필요하면 normalized 결과 파일도 저장
    if args.save_normalized:
        for row in rows:
            row["gold_label"] = normalize_label(row.get("gold_label"))

            if "direct" in row:
                row["direct"]["label"] = normalize_label(row["direct"].get("label"))

            for method in methods:
                if method == "direct":
                    continue
                if method in row:
                    row[method]["initial_label"] = normalize_label(
                        row[method].get("initial_label")
                    )
                    row[method]["refined_label"] = normalize_label(
                        row[method].get("refined_label")
                    )

        with open(args.save_normalized, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"\nSaved normalized file to {args.save_normalized}")


if __name__ == "__main__":
    main()