"""
rationale_refine_word_cand.py
===============================
Cue-based Candidate Rationale Self-Refinement.

[전체 흐름]

    Step 1. Candidate Extraction (NER + POS 필터)
            claim / evidence에서 의미 있는 단어/구만 후보로 추출.
            - NER  : 인물명, 지명, 기관, 날짜, 수치 → 반드시 포함
            - POS  : 명사, 동사, 형용사, 부사 (content word)
            - 관사/전치사/접속사 등 function word 제외
            → claim 20단어 기준 전체 masking 대비 약 1/3~1/4 수준으로 감소

    Step 2. Rationale Verification (Perturbation + Label Flip + Confidence Drop)
            각 후보를 [MASK]로 치환 후 LLM에 재질문.

            검증 기준 (우선순위 순):
            ① Label Flip       : label이 바뀌면 → 핵심 rationale 확정
               · flip 방향 기록
                 - initial→REFUTES : "flip_to_refutes" (핵심 대립 근거)
                 - initial→NEI     : "flip_to_nei"     (핵심 정보 근거)
            ② Confidence Drop  : label 동일, but 확신도가 threshold 이상 하락
               → "confidence_drop" (약하지만 관련 있는 근거)

            importance_score 계산:
                label_flip    → 1.0
                conf_drop     → drop_amount  (0.0~1.0)
                해당 없음      → 0.0

    Step 3. Top-K 선정 & Refinement
            importance_score 내림차순 → Top-K 선정
            검증된 rationale + 유형 정보를 refinement prompt에 전달.

[파일 의존성]
    llm_client_wordr.py  → call_llm, call_llm_with_logprobs
    prompts_word_cand.py → counterfactual_verify_prompt,
                           rationale_refine_prompt,
                           rationale_refine_prompt_no_rationale
    (NER/POS: spacy 있으면 사용, 없으면 regex fallback 자동 선택)
"""

import re
import math
from llm_client_wordr import call_llm, call_llm_with_logprobs
from prompts_word_cand import (
    counterfactual_verify_prompt,
    rationale_refine_prompt,
    rationale_refine_prompt_no_rationale,
)

# spacy 있으면 사용, 없으면 regex fallback
try:
    import spacy
    _nlp = spacy.load("en_core_web_sm")
    _USE_SPACY = True
except Exception:
    _USE_SPACY = False

VITAMC_CUE_PATTERNS = {
    "negation": [
        r"\bnot\b", r"\bnever\b", r"\bno\b", r"\bwithout\b",
        r"\bdid not\b", r"\bdoes not\b", r"\bdo not\b",
        r"\bfailed to\b", r"\bfails to\b",
    ],
    "directionality": [
        r"\bincrease(?:d|s|ing)?\b",
        r"\bdecrease(?:d|s|ing)?\b",
        r"\brise(?:n|s|ing)?\b",
        r"\braise(?:d|s|ing)?\b",
        r"\bfall(?:en|s|ing)?\b",
        r"\breduce(?:d|s|ing)?\b",
        r"\blower(?:ed|ing|s)?\b",
        r"\bhigher\b", r"\blower\b",
        r"\bmore\b", r"\bless\b",
        r"\bimprove(?:d|s|ing)?\b",
        r"\bworsen(?:ed|s|ing)?\b",
        r"\bpromote(?:d|s|ing)?\b",
        r"\binhibit(?:ed|s|ing)?\b",
        r"\bcause(?:d|s|ing)?\b",
        r"\bprevent(?:ed|s|ing)?\b",
    ],
    "temporal": [
        r"\bbefore\b", r"\bafter\b", r"\bduring\b",
        r"\bearlier\b", r"\blater\b",
        r"\bformer\b", r"\bcurrent\b",
        r"\bprevious\b", r"\bsubsequent\b",
        r"\bprior to\b", r"\bas of\b",
        r"\b\d{4}\b",
    ],
    "numerical": [
        r"\b\d+(?:\.\d+)?%?\b",
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b",
        r"\bmost\b", r"\bleast\b",
        r"\bmajority\b", r"\bminority\b",
        r"\bmore than\b", r"\bless than\b",
        r"\bat least\b", r"\bat most\b",
    ],
    "scope_condition": [
        r"\bonly\b", r"\ball\b", r"\bsome\b", r"\bmost\b",
        r"\bmany\b", r"\bfew\b",
        r"\bmay\b", r"\bcan\b", r"\bcould\b", r"\bmust\b",
        r"\bif\b", r"\bunless\b", r"\bwhen\b",
        r"\bunder\b", r"\bexcept\b",
    ],
    "relation": [
        r"\bborn in\b", r"\bdied in\b",
        r"\bcaused by\b", r"\bassociated with\b",
        r"\bmember of\b", r"\bpart of\b",
        r"\bowned by\b", r"\bfounded by\b", r"\bfounded in\b",
    ],
}


def extract_vitaminc_candidate_rationales(
    claim: str,
    evidence: str,
    include_claim: bool = True,
    max_candidates: int = 12,
) -> list[dict]:
    candidates = []

    def collect_from_text(text: str, source: str):
        for diff_type, patterns in VITAMC_CUE_PATTERNS.items():
            for pattern in patterns:
                for m in re.finditer(pattern, text, flags=re.IGNORECASE):
                    candidates.append({
                        "span": m.group(0).strip(),
                        "source": source,
                        "difference_type": diff_type,
                        "start": m.start(),
                        "end": m.end(),
                    })

    collect_from_text(evidence, "evidence")
    if include_claim:
        collect_from_text(claim, "claim")

    seen = set()
    unique = []
    for c in candidates:
        key = (
            c["source"],
            c["span"].lower(),
            c["difference_type"],
            c["start"],
            c["end"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    unique = sorted(
        unique,
        key=lambda x: (
            0 if x["source"] == "evidence" else 1,
            x["start"],
            x["end"],
        )
    )

    return unique[:max_candidates]

# ══════════════════════════════════════════════════════════════
# Parsing helpers
# ══════════════════════════════════════════════════════════════

def parse_label(output: str) -> str:
    match = re.search(
        r"Label:\s*\[?\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO|NEI)\s*\]?",
        output,
        re.IGNORECASE,
    )
    if match:
        label = match.group(1).upper().strip()
        return "NOT ENOUGH INFO" if label == "NEI" else label
    return "PARSE_ERROR"


# ══════════════════════════════════════════════════════════════
# Step 1: Candidate Extraction (NER + POS 필터)
# ══════════════════════════════════════════════════════════════

# ── regex fallback용 상수 ──────────────────────────────────────

# 제외할 function word (관사, 전치사, 접속사, 대명사 등)
_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must",
    "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "from", "to", "of", "up", "down",
    "and", "but", "or", "nor", "so", "yet",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "who", "what", "which", "when", "where", "how",
    "not", "no", "also", "than", "as", "if", "then",
}

# 수치/날짜 패턴 (NER 대용)
_NUM_DATE_PATTERN = re.compile(
    r"\b(\d{4}|\d{1,2}(?:st|nd|rd|th)?|\d+(?:\.\d+)?%?)\b"
    r"|(?:january|february|march|april|may|june|july|august|"
    r"september|october|november|december)\b",
    re.IGNORECASE,
)

# 대문자 시작 (고유명사 근사)
_PROPER_NOUN_PATTERN = re.compile(r"^[A-Z][a-z]+")


def _extract_candidates_regex(text: str) -> list[str]:
    """
    spacy 없을 때 regex 기반 후보 추출.
    대문자 시작 단어(고유명사 근사) + 수치/날짜 + stopword 제외 일반 단어.
    """
    tokens = re.findall(r"\b\w[\w'-]*\b", text)
    candidates = []
    seen = set()

    for tok in tokens:
        lower = tok.lower()
        if lower in seen:
            continue
        # 1글자 제외
        if len(tok) <= 1:
            continue
        # stopword 제외
        if lower in _STOPWORDS:
            continue
        # 수치/날짜 포함
        if _NUM_DATE_PATTERN.match(tok):
            candidates.append(tok)
            seen.add(lower)
            continue
        # 대문자 시작 (고유명사 근사)
        if _PROPER_NOUN_PATTERN.match(tok):
            candidates.append(tok)
            seen.add(lower)
            continue
        # 일반 content word (3글자 이상)
        if len(tok) >= 3:
            candidates.append(tok)
            seen.add(lower)

    return candidates


# spacy POS 태그 중 content word에 해당하는 것
_CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}

# spacy NER 타입
_NER_TYPES = {
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "DATE", "TIME", "MONEY", "QUANTITY", "CARDINAL", "ORDINAL",
    "EVENT", "WORK_OF_ART", "LAW", "PRODUCT",
}


def _extract_candidates_spacy(text: str) -> list[str]:
    """
    spacy 기반 NER + POS 필터 후보 추출.
    NER entity → 우선 포함 (multi-word span 그대로)
    나머지 content word → 개별 token으로 포함
    """
    doc = _nlp(text)
    candidates = []
    seen = set()

    # NER entity 먼저 (multi-word span 우선)
    for ent in doc.ents:
        if ent.label_ in _NER_TYPES:
            span_text = ent.text.strip()
            if span_text.lower() not in seen:
                candidates.append(span_text)
                seen.add(span_text.lower())

    # NER에 포함되지 않은 content word 추가
    ner_char_ranges = {(ent.start_char, ent.end_char) for ent in doc.ents}

    for token in doc:
        if token.pos_ not in _CONTENT_POS:
            continue
        if token.is_stop:
            continue
        if len(token.text) <= 1:
            continue
        # 이미 NER span에 포함된 토큰 skip
        in_ner = any(
            s <= token.idx < e for s, e in ner_char_ranges
        )
        if in_ner:
            continue
        lower = token.lemma_.lower()
        if lower not in seen:
            candidates.append(token.text)
            seen.add(lower)

    return candidates


def extract_candidates(text: str) -> list[str]:
    """
    spacy 있으면 NER+POS, 없으면 regex fallback.
    """
    if _USE_SPACY:
        return _extract_candidates_spacy(text)
    return _extract_candidates_regex(text)


# ══════════════════════════════════════════════════════════════
# Step 2: Rationale Verification
# ══════════════════════════════════════════════════════════════

def mask_span(text: str, span: str) -> str:
    """span을 [MASK]로 치환 (대소문자 무시, 첫 등장만)."""
    return re.sub(re.escape(span), "[MASK]", text, count=1, flags=re.IGNORECASE)


def verify_rationale(
    claim: str,
    evidence: str,
    initial_label: str,
    initial_conf: float | None,
    span: str,
    source: str,
    confidence_drop_threshold: float = 0.2,
) -> dict:
    """
    단일 span에 대한 perturbation 검증.

    검증 기준:
        ① Label Flip       → importance_score = 1.0
           · →REFUTES      : rationale_type = "flip_to_refutes"
           · →NEI           : rationale_type = "flip_to_nei"
        ② Confidence Drop  → importance_score = drop_amount (label 동일)
           · drop ≥ threshold : rationale_type = "confidence_drop"
        해당 없음           → importance_score = 0.0, rationale_type = None

    Returns: dict with keys:
        span, source, masked_claim, masked_evidence,
        cf_label, cf_conf,
        label_changed, conf_drop, rationale_type, importance_score,
        is_verified,   # True이면 Top-K 후보
        cf_output
    """
    # 마스킹 적용
    if source == "claim":
        masked_claim    = mask_span(claim, span)
        masked_evidence = evidence
    else:
        masked_claim    = claim
        masked_evidence = mask_span(evidence, span)

    # 실제 마스킹 여부 확인
    if masked_claim == claim and masked_evidence == evidence:
        return {
            "span": span, "source": source,
            "masked_claim": masked_claim, "masked_evidence": masked_evidence,
            "cf_label": "NOT_FOUND", "cf_conf": None,
            "label_changed": False, "conf_drop": 0.0,
            "rationale_type": None, "importance_score": 0.0,
            "is_verified": False,
            "cf_output": "span not found in text",
        }

    # LLM 호출 (logprobs 포함)
    prompt = counterfactual_verify_prompt(masked_claim, masked_evidence)
    cf_output, cf_conf, cf_label_token, _ = call_llm_with_logprobs(
        prompt,
        max_tokens=80,
    )
    cf_label = parse_label(cf_output)

    # ── ① Label Flip 판정 ────────────────────────────────
    label_changed = (cf_label != initial_label and cf_label not in ("PARSE_ERROR", "NOT_FOUND"))

    if label_changed:
        if cf_label == "REFUTES":
            rationale_type = "flip_to_refutes"
        else:
            rationale_type = "flip_to_nei"
        importance_score = 1.0
        conf_drop        = 0.0  # flip이면 conf_drop은 의미 없음

    else:
        # ── ② Confidence Drop 판정 ───────────────────────
        if initial_conf is not None and cf_conf is not None:
            conf_drop = initial_conf - cf_conf   # 양수면 확신도 하락
        else:
            conf_drop = 0.0

        if conf_drop >= confidence_drop_threshold:
            rationale_type   = "confidence_drop"
            importance_score = conf_drop          # drop 크기가 score
        else:
            rationale_type   = None
            importance_score = 0.0

    is_verified = importance_score > 0.0

    return {
        "span":             span,
        "source":           source,
        "masked_claim":     masked_claim,
        "masked_evidence":  masked_evidence,
        "cf_label":         cf_label,
        "cf_conf":          cf_conf,
        "label_changed":    label_changed,
        "conf_drop":        round(conf_drop, 4),
        "rationale_type":   rationale_type,
        "importance_score": round(importance_score, 4),
        "is_verified":      is_verified,
        "cf_output":        cf_output,
    }


# ══════════════════════════════════════════════════════════════
# Full Pipeline
# ══════════════════════════════════════════════════════════════

def wordcand_self_refinement(
    claim: str,
    evidence: str,
    initial_answer: str,
    initial_conf: float | None = None,
    top_k: int = 5,
    confidence_drop_threshold: float = 0.2,
    verbose: bool = True,
) -> dict:
    """
    Word-level Rationale 기반 Self-Refinement 전체 파이프라인.

    Args:
        claim                     : FEVER claim 문자열
        evidence                  : FEVER evidence 문자열
        initial_answer            : direct_prediction()의 raw LLM output
        top_k                     : 최종 rationale 수
        confidence_drop_threshold : confidence drop 기준 (dev set으로 탐색 권장)
                                    기본값 0.2 → ablation: 0.1 / 0.15 / 0.2 / 0.25 / 0.3
        verbose                   : 중간 과정 출력

    Returns: {
        "initial_label"              : str,
        "initial_conf"               : float | None,
        "extraction_method"          : "spacy" | "regex",

        "claim_candidates"           : list[str],
        "evidence_candidates"        : list[str],
        "total_candidates"           : int,

        "verification_results"       : list[dict],   # 전체 검증 결과
        "verified_rationales"        : list[dict],   # Top-K (is_verified=True)
        "verified_spans"             : list[str],    # span 문자열만

        "refined_answer"             : str,
        "refined_label"              : str,
        "num_verified"               : int,
        "label_changed_after_refine" : bool,
    }
    """
    # ── initial label & confidence 추출 ────────────────────────
    # _, initial_conf, _, _ = call_llm_with_logprobs(
        # initial_answer를 다시 파싱하는 게 아니라
        # initial_answer에서 label 확률을 뽑기 위해 raw output을 그대로 활용.
        # 단, 이미 생성된 텍스트이므로 logprob을 직접 재추출하기 위해
        # direct_prompt 결과의 label 확률을 별도로 받아야 함.
        # → run_experiment_wordr.py에서 initial_conf를 넘겨주는 방식으로 처리.
        # 여기선 fallback: initial_answer를 그대로 LLM에 넣어 재확인.
    #    f"Repeat this exactly:\n{initial_answer}",
    #    max_tokens=len(initial_answer.split()) + 20,
    # )
    initial_label = parse_label(initial_answer)

    if verbose:
        conf_str = f"{initial_conf:.3f}" if initial_conf is not None else "N/A"
        print(
            f"    [WordR] initial_label={initial_label}, initial_conf={conf_str}",
            flush=True,
        )

    # ── Step 1: Candidate Extraction ───────────────────────────
    candidate_rationales = extract_vitaminc_candidate_rationales(
        claim=claim,
        evidence=evidence,
        include_claim=True,
        max_candidates=12,
    )

    all_spans = [
        (c["span"], c["source"], c.get("difference_type", "unknown"))
        for c in candidate_rationales
    ]

    extraction_method = "vitaminc_cue"
    total_candidates = len(all_spans)

    # ── Step 2: Rationale Verification ─────────────────────────
    verification_results = []

    for i, (span, source, difference_type) in enumerate(all_spans, 1):
        result = verify_rationale(
            claim                     = claim,
            evidence                  = evidence,
            initial_label             = initial_label,
            initial_conf              = initial_conf,
            span                      = span,
            source                    = source,
            confidence_drop_threshold = confidence_drop_threshold,
        )
        result["difference_type"] = difference_type
        verification_results.append(result)

        if verbose:
            if result["is_verified"]:
                print(
                    f"    [WordR] ({i}/{total_candidates}) [{source}] '{span}' "
                    f"→ cf={result['cf_label']} | "
                    f"type={result['rationale_type']} | "
                    f"score={result['importance_score']:.3f} ✓",
                    flush=True,
                )
            else:
                print(
                    f"    [WordR] ({i}/{total_candidates}) [{source}] '{span}' "
                    f"→ cf={result['cf_label']} | score=0.000",
                    flush=True,
                )

    # ── Top-K 선정 ──────────────────────────────────────────────
    # importance_score 내림차순, 동점이면 span 길이 오름차순
    verified = [r for r in verification_results if r["is_verified"]]
    verified.sort(key=lambda x: (-x["importance_score"], len(x["span"])))
    top_k_rationales = verified[:top_k]
    verified_spans   = [r["span"] for r in top_k_rationales]

    if verbose:
        print(
            f"    [WordR] Verified={len(verified)} → "
            f"Top-{top_k}: {verified_spans}",
            flush=True,
        )

    # ── Step 3: Refinement ──────────────────────────────────────
    if top_k_rationales:
        refine_prompt = rationale_refine_prompt(
            claim=claim,
            evidence=evidence,
            initial_answer=initial_answer,
            verified_rationales=top_k_rationales,
            candidate_rationales=candidate_rationales,
    )
    
    else:
        refine_prompt = rationale_refine_prompt_no_rationale(
            claim=claim,
            evidence=evidence,
            initial_answer=initial_answer,
            candidate_rationales=candidate_rationales,
        )

    refined_answer = call_llm(refine_prompt, max_tokens=160)
    refined_label  = parse_label(refined_answer)

    return {
        "initial_label": initial_label,
        "initial_conf": round(initial_conf, 4) if initial_conf is not None else None,
        "extraction_method": extraction_method,

        "candidate_rationales": candidate_rationales,
        "total_candidates": total_candidates,

        "verification_results": verification_results,
        "verified_rationales": top_k_rationales,
        "verified_spans": verified_spans,

        "refined_answer": refined_answer,
        "refined_label": refined_label,
        "num_verified": len(top_k_rationales),
        "label_changed_after_refine": refined_label != initial_label,
    }