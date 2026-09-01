"""
prompts_wordr.py
================
Word-level Rationale 기반 Conservative Self-Refinement 전용 프롬프트.

이 파일은 아래 두 가지를 담당한다.

1. counterfactual_verify_prompt
   - span을 [MASK]로 치환한 claim/evidence를 받아 label 재예측
   - perturbation importance 측정에 사용

2. rationale_refine_prompt / rationale_refine_prompt_no_rationale
   - 검증된 rationale을 diagnostic signal로 사용
   - initial answer를 무조건 바꾸지 않고, 명확한 오류가 있을 때만 수정
"""


# ─────────────────────────────────────────────────────────────
# Step 1: Counterfactual 검증용
# ─────────────────────────────────────────────────────────────

def counterfactual_verify_prompt(masked_claim: str, masked_evidence: str) -> str:
    """
    span을 [MASK]로 치환한 claim/evidence를 받아 label을 재예측.
    마스킹된 부분을 알 수 없는 정보로 간주하도록 명시.
    """
    return f"""
You are a fact verification system.

Your task is to verify the claim using only the provided evidence.
Do not use outside knowledge.

Some parts of the text have been replaced with [MASK].
Treat [MASK] as information that is unknown or unavailable.
Make your judgment based only on the remaining visible content.

Classify the relationship as one of:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information.

Claim:
{masked_claim}

Evidence:
{masked_evidence}

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Explanation: ...
""".strip()


# ─────────────────────────────────────────────────────────────
# Step 2: Verified Rationale 기반 Conservative Refinement
# ─────────────────────────────────────────────────────────────

_RATIONALE_TYPE_DESC = {
    "flip_to_refutes": (
        "CONTRADICTION-SENSITIVE RATIONALE "
        "(masking this word/phrase caused the prediction to flip to REFUTES; "
        "use it only to check whether there is a real contradiction between claim and evidence)"
    ),
    "flip_to_nei": (
        "UNCERTAINTY-SENSITIVE RATIONALE "
        "(masking this word/phrase caused the prediction to flip to NOT ENOUGH INFO; "
        "this indicates that the word/phrase may be informative, but it is not by itself a reason to change the label)"
    ),
    "confidence_drop": (
        "CONFIDENCE-SENSITIVE RATIONALE "
        "(masking this word/phrase reduced confidence; "
        "this is a weak diagnostic signal and should not by itself justify changing the label)"
    ),
}


def _classify_signal_strength(verified_rationales: list[dict]) -> str:
    """
    verified_rationale 목록을 보고 신호 강도를 판정.

    강도 기준:
        STRONG  : flip_to_refutes가 1개 이상
                  → 명시적 모순 가능성, CHANGE 권고
        MODERATE: flip_to_nei가 evidence source에서 1개 이상
                  → evidence 핵심 정보 소실, 재검토 필요
        WEAK    : flip_to_nei가 claim source에서만 or confidence_drop만
                  → claim 핵심어 소실 또는 확신도 하락, KEEP 기본
    """
    types = [(r.get("rationale_type"), r.get("source")) for r in verified_rationales]

    if any(t == "flip_to_refutes" for t, _ in types):
        return "STRONG"
    if any(t == "flip_to_nei" and s == "evidence" for t, s in types):
        return "MODERATE"
    return "WEAK"


def rationale_refine_prompt(claim, evidence, initial_answer, rationales):
    rationale_lines = []

    for r in rationales:
        span = r.get("span", "")
        source = r.get("source", "")
        rtype = r.get("rationale_type", "")
        score = r.get("importance_score", 0.0)
        cf_label = r.get("cf_label", "")

        rationale_lines.append(
            f"- span: {span}\n"
            f"  source: {source}\n"
            f"  rationale_type: {rtype}\n"
            f"  counterfactual_label: {cf_label}\n"
            f"  importance_score: {score:.3f}"
        )

    rationale_text = "\n".join(rationale_lines) if rationale_lines else "None"

    return f"""
You are refining a fact verification answer.

Task:
Given a claim and evidence, decide whether the claim is SUPPORTS, REFUTES, or NOT ENOUGH INFO.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

Verified word-level rationales:
{rationale_text}

Important rules:
1. Verified rationales are sensitivity signals, not direct proof that the previous answer is wrong.
2. Do not change the label only because masking a word changed the prediction.
3. Do not change the label only because a word is important.
4. Change the label only if the rationales reveal a clear factual mismatch, such as:
   - contradiction
   - negation mismatch
   - wrong date or time
   - wrong number or quantity
   - wrong entity
   - missing required condition
5. If the evidence still supports the previous answer, KEEP the previous label.
6. If there is no clear and explicit reason to change the previous label, KEEP it.

Be conservative. Prefer KEEP unless the previous label is clearly wrong.

Return exactly in this format:
Label: SUPPORTS / REFUTES / NOT ENOUGH INFO
Decision: KEEP / CHANGE
Explanation: ...
""".strip()


def rationale_refine_prompt_no_rationale(claim, evidence, initial_answer):
    return f"""
You are refining a fact verification answer.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

No verified word-level rationales were found.

Important rules:
1. Since there are no verified rationales, do not change the previous label unless it is clearly and explicitly wrong.
2. Prefer KEEP.
3. Only change the label if the previous answer contains an obvious contradiction with the evidence.

Return exactly in this format:
Label: SUPPORTS / REFUTES / NOT ENOUGH INFO
Decision: KEEP / CHANGE
Explanation: ...
""".strip()