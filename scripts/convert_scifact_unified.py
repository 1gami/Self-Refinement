import argparse
import json
import os
from collections import Counter

LABEL_MAP = {
    "SUPPORT": "SUPPORT",
    "SUPPORTED": "SUPPORT",
    "SUPPORTS": "SUPPORT",
    "CONTRADICT": "REFUTE",
    "CONTRADICTS": "REFUTE",
    "CONTRADICTION": "REFUTE",
    "REFUTE": "REFUTE",
    "REFUTES": "REFUTE",
    "REFUTED": "REFUTE",
    "NEI": "NOT ENOUGH INFO",
    "NOT ENOUGH INFO": "NOT ENOUGH INFO",
    "NOT_ENOUGH_INFO": "NOT ENOUGH INFO",
}


def normalize_label(label: str) -> str:
    if label is None:
        return "NOT ENOUGH INFO"
    key = str(label).strip().upper().replace("_", " ")
    if key not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")
    return LABEL_MAP[key]


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(rows, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_corpus(corpus_path):
    corpus = {}
    for doc in read_jsonl(corpus_path):
        doc_id = str(doc["doc_id"])
        abstract = doc.get("abstract", [])
        if isinstance(abstract, str):
            abstract = [abstract]
        corpus[doc_id] = {
            "title": doc.get("title", ""),
            "abstract": [str(s).strip() for s in abstract if str(s).strip()],
        }
    return corpus


def make_evidence_text(doc, sentence_indices=None, mode="gold_only"):
    title = doc.get("title", "").strip()
    abstract = doc.get("abstract", [])

    if mode == "full_abstract" or sentence_indices is None:
        selected = abstract
    else:
        selected = []
        for idx in sentence_indices:
            if isinstance(idx, int) and 0 <= idx < len(abstract):
                selected.append(abstract[idx])

    body = " ".join(selected).strip()
    if title and body:
        return f"Title: {title}\nAbstract: {body}"
    if title:
        return f"Title: {title}"
    return body


def convert_scifact(claims_path, corpus_path, output_path, split, evidence_mode="gold_only", include_nei=True):
    """
    evidence_mode:
      - gold_only: SUPPORT/REFUTE examples use gold rationale sentences only.
      - full_abstract: SUPPORT/REFUTE examples use full abstract of the cited document.

    include_nei:
      - True: claims with empty evidence and cited_doc_ids are converted to NOT ENOUGH INFO examples
              using the full cited abstract as evidence.
      - False: skip those NEI examples.
    """
    corpus = load_corpus(corpus_path)
    rows = []
    skipped = Counter()

    for claim_ex in read_jsonl(claims_path):
        claim_id = claim_ex.get("id")
        claim = str(claim_ex.get("claim", "")).strip()
        if not claim:
            skipped["empty_claim"] += 1
            continue

        evidence_dict = claim_ex.get("evidence", {}) or {}
        cited_doc_ids = [str(d) for d in claim_ex.get("cited_doc_ids", [])]

        # SUPPORT / REFUTE examples from annotated evidence sets
        for doc_id, ev_sets in evidence_dict.items():
            doc_id = str(doc_id)
            doc = corpus.get(doc_id)
            if doc is None:
                skipped["missing_doc"] += 1
                continue

            for ev_idx, ev in enumerate(ev_sets):
                sent_ids = ev.get("sentences", [])
                label = normalize_label(ev.get("label"))
                evidence_text = make_evidence_text(doc, sent_ids, mode=evidence_mode)

                if not evidence_text:
                    skipped["empty_evidence"] += 1
                    continue

                rows.append({
                    "id": f"scifact_{split}_{claim_id}_{doc_id}_{ev_idx}",
                    "dataset": "scifact",
                    "claim": claim,
                    "evidence": evidence_text,
                    "label": label,
                    "metadata": {
                        "split": split,
                        "claim_id": claim_id,
                        "doc_id": doc_id,
                        "evidence_set_index": ev_idx,
                        "sentence_indices": sent_ids,
                        "original_label": ev.get("label"),
                        "evidence_mode": evidence_mode,
                    },
                })

        # NEI examples: empty evidence field but cited documents exist
        if include_nei and not evidence_dict and cited_doc_ids:
            for doc_id in cited_doc_ids:
                doc = corpus.get(doc_id)
                if doc is None:
                    skipped["missing_doc_nei"] += 1
                    continue
                evidence_text = make_evidence_text(doc, None, mode="full_abstract")
                if not evidence_text:
                    skipped["empty_nei_evidence"] += 1
                    continue
                rows.append({
                    "id": f"scifact_{split}_{claim_id}_{doc_id}_nei",
                    "dataset": "scifact",
                    "claim": claim,
                    "evidence": evidence_text,
                    "label": "NOT ENOUGH INFO",
                    "metadata": {
                        "split": split,
                        "claim_id": claim_id,
                        "doc_id": doc_id,
                        "source": "empty_evidence_with_cited_doc",
                        "evidence_mode": "full_abstract",
                    },
                })

    write_jsonl(rows, output_path)
    dist = Counter(row["label"] for row in rows)
    print(f"Saved {len(rows)} examples to {output_path}")
    print("Label distribution:", dict(dist))
    if skipped:
        print("Skipped:", dict(skipped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--evidence-mode", choices=["gold_only", "full_abstract"], default="gold_only")
    parser.add_argument("--no-nei", action="store_true", help="Do not create NOT ENOUGH INFO examples from empty-evidence cited docs")
    args = parser.parse_args()

    convert_scifact(
        claims_path=args.claims,
        corpus_path=args.corpus,
        output_path=args.output,
        split=args.split,
        evidence_mode=args.evidence_mode,
        include_nei=not args.no_nei,
    )
