import json
import argparse
from pathlib import Path


REFINE_METHODS = [
    "basic_self_refine",
    "evidence_aware_self_refine",
    "vitaminc_style_self_refine",
    "wordr_self_refine",
]


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_pred(row, method):
    if method == "direct":
        return row.get("direct", {}).get("label")

    return row.get(method, {}).get("refined_label")


def get_answer(row, method):
    if method == "direct":
        return row.get("direct", {}).get("answer", "")

    return row.get(method, {}).get("refined_answer", "")


def available_methods(row):
    methods = []

    if "direct" in row:
        methods.append("direct")

    for method in REFINE_METHODS:
        if method in row:
            methods.append(method)

    return methods


def compact_case(row, methods):
    """
    사람이 보기 쉽게 필요한 정보만 정리.
    """
    gold = row.get("gold_label")

    predictions = {}
    answers = {}

    for method in methods:
        predictions[method] = get_pred(row, method)
        answers[method] = get_answer(row, method)

    item = {
        "id": row.get("id"),
        "claim": row.get("claim"),
        "evidence": row.get("evidence"),
        "gold_label": gold,
        "predictions": predictions,
        "answers": answers,
        "source": row.get("source", ""),
        "comment": row.get("comment", ""),
    }

    if "wordr_self_refine" in row:
        wordr = row["wordr_self_refine"]
        item["wordr"] = {
            "verified_spans": wordr.get("verified_spans", []),
            "verified_rationales": wordr.get("verified_rationales", []),
            "num_verified": wordr.get("num_verified", 0),
            "claim_candidates": wordr.get("claim_candidates", []),
            "evidence_candidates": wordr.get("evidence_candidates", []),
        }

    return item


def extract_all_methods_wrong(results):
    """
    모든 method의 prediction이 gold와 다른 case.
    direct, basic, evidence-aware, vitaminc, wordr 중
    결과 파일에 존재하는 method만 대상으로 함.
    """
    extracted = []

    for row in results:
        gold = row.get("gold_label")
        methods = available_methods(row)

        if not methods:
            continue

        preds = [get_pred(row, method) for method in methods]

        # 모든 method가 gold와 다르게 예측한 경우
        if all(pred != gold for pred in preds):
            extracted.append(compact_case(row, methods))

    return extracted


def extract_direct_wrong_refine_correct(results, method):
    """
    Direct는 틀렸는데 특정 refinement method가 맞게 고친 case.
    W→C
    """
    extracted = []

    for row in results:
        if method not in row:
            continue

        gold = row.get("gold_label")
        direct_pred = get_pred(row, "direct")
        refine_pred = get_pred(row, method)

        if direct_pred != gold and refine_pred == gold:
            item = compact_case(row, ["direct", method])
            item["transition"] = "direct_wrong_to_refine_correct"
            item["method"] = method
            extracted.append(item)

    return extracted


def extract_direct_correct_refine_wrong(results, method):
    """
    Direct는 맞았는데 특정 refinement method가 틀리게 바꾼 case.
    C→W
    """
    extracted = []

    for row in results:
        if method not in row:
            continue

        gold = row.get("gold_label")
        direct_pred = get_pred(row, "direct")
        refine_pred = get_pred(row, method)

        if direct_pred == gold and refine_pred != gold:
            item = compact_case(row, ["direct", method])
            item["transition"] = "direct_correct_to_refine_wrong"
            item["method"] = method
            extracted.append(item)

    return extracted


def extract_all_refinement_transitions(results):
    """
    각 refinement method별 W→C, C→W를 모두 추출.
    """
    output = {}

    for method in REFINE_METHODS:
        # 결과 파일에 해당 method가 없으면 skip
        if not results or method not in results[0]:
            continue

        output[method] = {
            "direct_wrong_refine_correct": extract_direct_wrong_refine_correct(
                results, method
            ),
            "direct_correct_refine_wrong": extract_direct_correct_refine_wrong(
                results, method
            ),
        }

    return output


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
        default="results/extracted_cases",
        help="Output directory",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Also save each extracted set as jsonl",
    )

    args = parser.parse_args()

    results = read_jsonl(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_stem = Path(args.input).stem

    print(f"Loaded {len(results)} rows from {args.input}")

    # 1. 모든 method가 틀린 case
    all_wrong = extract_all_methods_wrong(results)

    all_wrong_path = out_dir / f"{input_stem}_all_methods_wrong.json"
    write_json(all_wrong_path, all_wrong)

    print(f"All methods wrong: {len(all_wrong)}")
    print(f"Saved to {all_wrong_path}")

    if args.jsonl:
        write_jsonl(
            out_dir / f"{input_stem}_all_methods_wrong.jsonl",
            all_wrong,
        )

    # 2. method별 W→C, C→W
    transitions = extract_all_refinement_transitions(results)

    transition_summary = {}

    for method, cases in transitions.items():
        wc = cases["direct_wrong_refine_correct"]
        cw = cases["direct_correct_refine_wrong"]

        transition_summary[method] = {
            "direct_wrong_refine_correct_count": len(wc),
            "direct_correct_refine_wrong_count": len(cw),
        }

        wc_path = out_dir / f"{input_stem}_{method}_direct_wrong_refine_correct.json"
        cw_path = out_dir / f"{input_stem}_{method}_direct_correct_refine_wrong.json"

        write_json(wc_path, wc)
        write_json(cw_path, cw)

        print(f"\n[{method}]")
        print(f"  Direct wrong → refine correct: {len(wc)}")
        print(f"  Saved to {wc_path}")
        print(f"  Direct correct → refine wrong: {len(cw)}")
        print(f"  Saved to {cw_path}")

        if args.jsonl:
            write_jsonl(
                out_dir / f"{input_stem}_{method}_direct_wrong_refine_correct.jsonl",
                wc,
            )
            write_jsonl(
                out_dir / f"{input_stem}_{method}_direct_correct_refine_wrong.jsonl",
                cw,
            )

    # 3. summary 저장
    summary = {
        "input": args.input,
        "num_samples": len(results),
        "all_methods_wrong_count": len(all_wrong),
        "transition_summary": transition_summary,
    }

    summary_path = out_dir / f"{input_stem}_extraction_summary.json"
    write_json(summary_path, summary)

    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()