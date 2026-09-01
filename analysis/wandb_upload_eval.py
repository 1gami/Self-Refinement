import json
import argparse
from pathlib import Path

import wandb


LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_metrics(summary):
    """
    evaluation_summary.json을 W&B scalar metric으로 펼침.
    """
    metrics = {}

    metrics["num_samples"] = summary.get("num_samples", 0)

    # gold distribution
    for label, count in summary.get("gold_distribution", {}).items():
        safe_label = label.replace(" ", "_")
        metrics[f"gold_dist/{safe_label}"] = count

    # method-level metrics
    for method, m in summary.get("methods", {}).items():
        metrics[f"{method}/accuracy"] = m.get("accuracy")
        metrics[f"{method}/macro_f1"] = m.get("macro_f1")
        metrics[f"{method}/parse_error_rate"] = m.get("parse_error_rate")

        # prediction distribution
        for label, count in m.get("prediction_distribution", {}).items():
            safe_label = label.replace(" ", "_")
            metrics[f"{method}/pred_dist/{safe_label}"] = count

        # per-class metrics
        for label, vals in m.get("per_class", {}).items():
            safe_label = label.replace(" ", "_")
            metrics[f"{method}/per_class/{safe_label}/precision"] = vals.get("precision")
            metrics[f"{method}/per_class/{safe_label}/recall"] = vals.get("recall")
            metrics[f"{method}/per_class/{safe_label}/f1"] = vals.get("f1")
            metrics[f"{method}/per_class/{safe_label}/support"] = vals.get("support")

        # refinement metrics
        if "refinement" in m:
            r = m["refinement"]
            for k, v in r.items():
                metrics[f"{method}/refinement/{k}"] = v

    # WordR metrics가 있을 경우
    if "wordr_rationale_metrics" in summary:
        w = summary["wordr_rationale_metrics"]
        for k, v in w.items():
            if isinstance(v, (int, float)):
                metrics[f"wordr/{k}"] = v

        for rtype, count in w.get("rationale_type_distribution", {}).items():
            metrics[f"wordr/rationale_type_dist/{rtype}"] = count

    # None 제거
    metrics = {k: v for k, v in metrics.items() if v is not None}

    return metrics


def make_method_table(summary):
    columns = [
        "method",
        "accuracy",
        "macro_f1",
        "parse_error_rate",
        "SUPPORTS_f1",
        "REFUTES_f1",
        "NEI_f1",
    ]

    data = []

    for method, m in summary.get("methods", {}).items():
        per_class = m.get("per_class", {})

        supports_f1 = per_class.get("SUPPORTS", {}).get("f1")
        refutes_f1 = per_class.get("REFUTES", {}).get("f1")
        nei_f1 = per_class.get("NOT ENOUGH INFO", {}).get("f1")

        data.append([
            method,
            m.get("accuracy"),
            m.get("macro_f1"),
            m.get("parse_error_rate"),
            supports_f1,
            refutes_f1,
            nei_f1,
        ])

    return wandb.Table(columns=columns, data=data)


def make_refinement_table(summary):
    columns = [
        "method",
        "C_to_C",
        "C_to_W",
        "W_to_C",
        "W_to_W",
        "correction_rate",
        "degradation_rate",
        "net_improvement",
        "net_improvement_rate",
        "label_changed",
        "label_change_accuracy",
    ]

    data = []

    for method, m in summary.get("methods", {}).items():
        if method == "direct":
            continue

        r = m.get("refinement")
        if not r:
            continue

        data.append([
            method,
            r.get("C_to_C"),
            r.get("C_to_W"),
            r.get("W_to_C"),
            r.get("W_to_W"),
            r.get("correction_rate"),
            r.get("degradation_rate"),
            r.get("net_improvement"),
            r.get("net_improvement_rate"),
            r.get("label_changed"),
            r.get("label_change_accuracy"),
        ])

    return wandb.Table(columns=columns, data=data)


def make_confusion_matrix_tables(summary):
    tables = {}

    for method, m in summary.get("methods", {}).items():
        cm = m.get("confusion_matrix", {})

        columns = ["gold_label", "pred_label", "count"]
        data = []

        for gold, pred_dict in cm.items():
            for pred, count in pred_dict.items():
                data.append([gold, pred, count])

        tables[f"confusion_matrix/{method}"] = wandb.Table(
            columns=columns,
            data=data,
        )

    return tables


def make_wordr_table(summary):
    if "wordr_rationale_metrics" not in summary:
        return None

    w = summary["wordr_rationale_metrics"]

    columns = ["metric", "value"]
    data = []

    for k, v in w.items():
        if isinstance(v, (int, float, str)):
            data.append([k, v])

    for rtype, count in w.get("rationale_type_distribution", {}).items():
        data.append([f"rationale_type/{rtype}", count])

    return wandb.Table(columns=columns, data=data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--project", type=str, default="wordr-self-refinement")
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)

    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="FEVER")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    args = parser.parse_args()

    summary = load_json(args.summary)

    run_name = args.run_name
    if run_name is None:
        run_name = Path(args.summary).stem

    config = {
        "summary_file": args.summary,
        "model": args.model,
        "dataset": args.dataset,
        "top_k": args.top_k,
        "confidence_drop_threshold": args.threshold,
        "num_samples": summary.get("num_samples"),
        "gold_distribution": summary.get("gold_distribution"),
    }

    with wandb.init(
        project=args.project,
        entity=args.entity,
        name=run_name,
        group=args.group,
        config=config,
    ) as run:

        # 1. scalar metrics
        metrics = flatten_metrics(summary)
        run.log(metrics)

        # 2. summary에도 핵심 지표 저장
        for k, v in metrics.items():
            run.summary[k] = v

        # 3. tables
        run.log({
            "tables/method_summary": make_method_table(summary),
            "tables/refinement_summary": make_refinement_table(summary),
        })

        cm_tables = make_confusion_matrix_tables(summary)
        for name, table in cm_tables.items():
            run.log({name: table})

        wordr_table = make_wordr_table(summary)
        if wordr_table is not None:
            run.log({"tables/wordr_rationale_summary": wordr_table})

        # 4. 원본 summary json도 artifact로 저장
        artifact = wandb.Artifact(
            name=f"{run_name}_evaluation_summary",
            type="evaluation",
        )
        artifact.add_file(args.summary)
        run.log_artifact(artifact)

        print(f"Uploaded to W&B project: {args.project}")
        print(f"Run name: {run_name}")


if __name__ == "__main__":
    main()