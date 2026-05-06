"""
prompts_word_cand.py
====================
VitaminC-style Candidate Rationale + Verified Word-level Rationale 기반
Conservative Self-Refinement 전용 프롬프트.

기존 prompts_wordr.py와의 차이:
- prompts_wordr.py:
    verified_rationales만 refinement prompt에 전달

- prompts_word_cand.py:
    1) VitaminC-style candidate rationales
       - entity / negation / numerical / temporal / directionality / scope / relation cue
       - "어디를 집중해서 봐야 하는지" 알려주는 inspection target

    2) verified rationales
       - masking perturbation으로 label flip 또는 confidence drop이 확인된 rationale
       - candidate보다 강한 diagnostic signal

두 정보를 모두 prompt에 전달하되,
최종 label은 항상 original unmasked evidence 기준으로 결정한다.
"""


# ─────────────────────────────────────────────────────────────
# Step 1: Counterfactual 검증용
# ─────────────────────────────────────────────────────────────

def counterfactual_verify_prompt(masked_claim: str, masked_evidence: str) -> str:
    """
    span을 [MASK]로 치환한 claim/evidence를 받아 label을 재예측.
    마스킹된 부분은 unknown/unavailable information으로 간주한다.
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
# Rationale type 설명
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


_CANDIDATE_TYPE_DESC = {
    "entity": (
        "ENTITY CUE "
        "(person, organization, location, object, event, disease, method, or other named concept)"
    ),
    "negation": (
        "NEGATION CUE "
        "(not, no, never, without, failed to, did not, etc.)"
    ),
    "numerical": (
        "NUMERICAL CUE "
        "(number, percentage, quantity, ranking, measurement, or comparison)"
    ),
    "temporal": (
        "TEMPORAL CUE "
        "(date, year, before/after, earlier/later, former/current, temporal condition)"
    ),
    "directionality": (
        "DIRECTIONALITY CUE "
        "(increase vs decrease, improve vs worsen, promote vs inhibit, cause vs prevent)"
    ),
    "scope_condition": (
        "SCOPE OR CONDITION CUE "
        "(only, all, some, most, may, can, must, if, unless, under certain conditions)"
    ),
    "relation": (
        "RELATION CUE "
        "(born in vs died in, caused by vs associated with, member of vs opponent of, etc.)"
    ),
    "unknown": (
        "POSSIBLE VITAMINC-STYLE CUE "
        "(a potentially label-changing word or phrase)"
    ),
}


def _format_candidate_rationales(candidate_rationales: list[dict] | None) -> str:
    """
    candidate_rationales 예시:
        {
            "span": "decreased",
            "source": "evidence",
            "difference_type": "directionality",
            "start": 10,
            "end": 19
        }
    """
    if not candidate_rationales:
        return "  - None"

    lines = []
    for r in candidate_rationales:
        span = r.get("span", "")
        source = r.get("source", "evidence")
        diff_type = r.get("difference_type", "unknown")
        desc = _CANDIDATE_TYPE_DESC.get(diff_type, _CANDIDATE_TYPE_DESC["unknown"])

        lines.append(
            f'  - "{span}" (from {source}) → VitaminC-style {desc}'
        )

    return "\n".join(lines)


def _format_verified_rationales(verified_rationales: list[dict] | None) -> str:
    """
    verified_rationales 예시:
        {
            "span": "decreased",
            "source": "evidence",
            "rationale_type": "flip_to_refutes",
            "importance_score": 0.72
        }
    """
    if not verified_rationales:
        return "  - None"

    lines = []
    for r in verified_rationales:
        span = r.get("span", "")
        source = r.get("source", "evidence")
        rtype = r.get("rationale_type", "confidence_drop")
        desc = _RATIONALE_TYPE_DESC.get(rtype, rtype)
        score = r.get("importance_score", None)

        if score is None:
            lines.append(f'  - "{span}" (from {source}) → {desc}')
        else:
            lines.append(f'  - "{span}" (from {source}, score={score}) → {desc}')

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Step 2: Candidate + Verified Rationale 기반 Conservative Refinement
# ─────────────────────────────────────────────────────────────

def rationale_refine_prompt(
    claim: str,
    evidence: str,
    initial_answer: str,
    verified_rationales: list[dict],
    candidate_rationales: list[dict] | None = None,
) -> str:
    """
    VitaminC-style candidate rationales와 verified rationales를 함께 전달하는
    conservative refinement prompt.

    candidate_rationales:
        - VitaminC-style cue detector가 뽑은 후보
        - 아직 perturbation으로 검증되지 않은 inspection target

    verified_rationales:
        - masking perturbation으로 label flip 또는 confidence drop이 확인된 rationale
        - stronger diagnostic signal
    """
    candidate_block = _format_candidate_rationales(candidate_rationales)
    verified_block = _format_verified_rationales(verified_rationales)

    return f"""
You are performing conservative rationale-guided self-refinement for fact verification.

Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to decide whether the previous answer should be kept or changed.

Labels:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information to determine whether the claim is true or false.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

VitaminC-style candidate rationales:
These are possible label-changing cues selected because VitaminC-style fact verification is sensitive to subtle differences in entities, negation, numbers, time, directionality, scope, and relations.
They are inspection targets, not verified explanations.

{candidate_block}

Verified word-level rationales:
These were verified by perturbation analysis.
They are stronger diagnostic signals than candidates, but they are still not guaranteed explanations.
The final label must be based on the original unmasked evidence.

{verified_block}

How to use these signals:
- Candidate rationales tell you where to inspect the claim and evidence carefully.
- Verified rationales tell you which words or phrases affected the model prediction under masking.
- Do not change the label solely because a candidate or verified rationale exists.
- Change the label only if the original unmasked evidence clearly supports a different label.
- Treat flip_to_nei and confidence_drop as uncertainty signals, not automatic reasons to choose NOT ENOUGH INFO.
- Treat flip_to_refutes as a contradiction signal only if the original evidence explicitly contradicts the claim.

Revision policy:
- Prefer KEEP unless there is a clear evidence-based error in the previous answer.
- CHANGE only when the claim-evidence relationship clearly supports a different label.
- Do not change to NOT ENOUGH INFO merely because masking caused uncertainty.
- Do not change to NOT ENOUGH INFO merely because a candidate rationale looks important.
- Choose NOT ENOUGH INFO only when the original evidence is insufficient to support or refute the claim.
- If the original evidence explicitly entails the claim, choose SUPPORTS.
- If the original evidence explicitly contradicts the claim, choose REFUTES.

Now decide whether to KEEP or CHANGE the previous answer.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Decision: [KEEP / CHANGE]
Used candidate rationales: [...]
Used verified rationales: [...]
Explanation: ...
""".strip()


def rationale_refine_prompt_no_rationale(
    claim: str,
    evidence: str,
    initial_answer: str,
    candidate_rationales: list[dict] | None = None,
) -> str:
    """
    verified rationale이 없을 때 fallback.
    candidate rationales는 inspection target으로만 사용한다.
    """
    candidate_block = _format_candidate_rationales(candidate_rationales)

    return f"""
You are performing conservative rationale-guided self-refinement for fact verification.

Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to decide whether the previous answer should be kept or changed.

Labels:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information to determine whether the claim is true or false.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

VitaminC-style candidate rationales:
These are possible label-changing cues selected from the original claim/evidence.
They are only inspection targets, not verified explanations.

{candidate_block}

Note:
Perturbation analysis found no verified word-level rationale whose removal significantly changes the predicted label or confidence.
This means there is no strong perturbation-based evidence suggesting that the previous answer should be revised.

How to use the candidates:
- Use them only to inspect the original evidence more carefully.
- Do not change the label solely because a candidate rationale exists.
- Do not change to NOT ENOUGH INFO merely because a candidate looks important.

Revision policy:
- Prefer KEEP.
- CHANGE the label only if the original claim and evidence clearly show that the previous answer is wrong.
- Choose NOT ENOUGH INFO only when the original evidence is insufficient to support or refute the claim.
- If the original evidence explicitly entails the claim, choose SUPPORTS.
- If the original evidence explicitly contradicts the claim, choose REFUTES.

Now decide whether to KEEP or CHANGE the previous answer.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Decision: [KEEP / CHANGE]
Used candidate rationales: [...]
Used verified rationales: []
Explanation: ...
""".strip()