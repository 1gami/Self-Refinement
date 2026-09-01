def direct_prompt(claim, evidence):
    return f"""
You are a fact verification system.

Your task is to verify the claim using only the provided evidence.
Do not use outside knowledge.

Classify the relationship between the claim and the evidence as one of the following labels:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information to determine whether the claim is true or false.

Claim:
{claim}

Evidence:
{evidence}

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Explanation: ...
""".strip()


# ─────────────────────────────────────────────────────────────
# Basic Self-Refine: Iteration-aware Conservative version
# ─────────────────────────────────────────────────────────────

def _iteration_text(iteration):
    if iteration is None:
        return ""
    return f"This is refinement iteration {iteration}."


def basic_feedback_prompt(claim, evidence, initial_answer, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are providing self-feedback for a fact verification answer.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

Your task is to critically review the previous answer.

Check whether:
1. The predicted label is correct based on the evidence.
2. The explanation is faithful to the evidence.
3. The answer relies on unsupported assumptions or outside knowledge.
4. The evidence actually supports, refutes, or is insufficient for the claim.
5. A different label would be better justified by the evidence.

Important:
- If the previous answer is already correct, explain why no revision is needed.
- If the previous answer is wrong or weakly justified, explain what should be revised and why.
- If this is not the first refinement iteration, do not suggest another label change unless a new clear evidence-based error is found.
- Do not suggest changing the label just to make the answer different.
- Be conservative: prefer keeping the previous label unless the evidence clearly requires a correction.

Do not produce the final revised answer yet.
Provide constructive feedback that can help improve the answer.

Answer strictly in the following format:
Feedback:
- Label assessment: ...
- Evidence assessment: ...
- Potential issue: ...
- Suggested revision: ...
""".strip()


def basic_refine_prompt(claim, evidence, initial_answer, feedback, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are refining a fact verification answer using self-feedback.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

Self-feedback:
{feedback}

Your task is to produce an improved final answer.

Revision policy:
- KEEP the previous label if it is well supported by the evidence.
- CHANGE the label only if the feedback identifies a clear evidence-based error.
- If this is not the first refinement iteration, avoid changing the label again unless a new clear evidence-based error is found.
- Do not change the label just to make the answer different.
- Do not change to NOT ENOUGH INFO merely because the evidence is short.

The final label must be one of:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information to determine whether the claim is true or false.

Base the final answer only on the evidence and the self-feedback.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Decision: [KEEP / CHANGE]
Explanation: ...
""".strip()


# ─────────────────────────────────────────────────────────────
# Evidence-aware Self-Refine: Iteration-aware Conservative version
# ─────────────────────────────────────────────────────────────

def evidence_aware_feedback_prompt(claim, evidence, initial_answer, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are a careful fact verification reviewer.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

Your task is to check whether the previous answer is clearly wrong.
Do not suggest changing the label unless there is an explicit evidence-based reason.

Review the previous answer by explicitly checking:

1. Does the evidence directly support the claim?
2. Is the directionality correct? For example, increase vs decrease, cause vs effect, promote vs inhibit.
3. Are the entities exactly the same?
4. Are quantities, dates, or conditions different?
5. Does the claim overstate the evidence?
6. Is the evidence merely related but insufficient?
7. Is there any negation or contrast that changes the verdict?
8. Is the previous label clearly inconsistent with the evidence?

Important:
- Prefer KEEP if the previous label is consistent with the evidence.
- Recommend CHANGE only if a clear verification error is found.
- If this is not the first refinement iteration, recommend CHANGE only when a new clear evidence-based error is found.
- Do not recommend NOT ENOUGH INFO merely because the evidence is short.
- Do not suggest changing the label just to make the answer different.

Do not produce the final answer.

Answer strictly in the following format:
Decision: [KEEP / CHANGE]
Error type: [none / entity_mismatch / directionality_error / numerical_mismatch / temporal_mismatch / negation_missed / overclaiming / insufficient_evidence / wrong_label]
Feedback: ...
""".strip()


def evidence_aware_refine_prompt(claim, evidence, initial_answer, feedback, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are performing evidence-aware self-refinement for fact verification.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to verify whether the previous answer should be kept or changed.

Revision policy:
- KEEP the previous label unless the feedback identifies a clear evidence-based error.
- CHANGE the label only when the claim-evidence relationship clearly supports a different label.
- If this is not the first refinement iteration, avoid changing the label again unless a new clear evidence-based error is found.
- Do not change the label just to make the answer different.
- Do not change to NOT ENOUGH INFO merely because the evidence is short.
- Choose NOT ENOUGH INFO only when the evidence is genuinely insufficient.
- If the evidence explicitly contradicts the claim, choose REFUTES.
- If the evidence explicitly entails the claim, choose SUPPORTS.

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

Evidence-aware feedback:
{feedback}

Now decide whether to KEEP or CHANGE the previous answer.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Decision: [KEEP / CHANGE]
Explanation: ...
""".strip()


# ─────────────────────────────────────────────────────────────
# VitaminC-style Self-Refine: Iteration-aware Conservative version
# ─────────────────────────────────────────────────────────────

def vitaminc_style_feedback_prompt(claim, evidence, initial_answer, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are a fact verification reviewer trained to detect subtle evidence changes.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Previous answer:
{initial_answer}

Your task is to check whether subtle evidence differences make the previous answer clearly wrong.

Focus on:
- entity mismatch
- negation
- numerical mismatch
- temporal mismatch
- causal direction
- increase/decrease direction
- condition or scope mismatch
- whether the evidence is related but not sufficient
- whether the claim requires stronger evidence than provided

Important revision policy:
- Do not suggest changing the label unless a subtle evidence difference clearly changes the verdict.
- Prefer KEEP if the previous label is consistent with the evidence.
- If this is not the first refinement iteration, recommend CHANGE only when a new clear evidence-based error is found.
- Do not change to NOT ENOUGH INFO merely because the evidence is short or indirect.
- Choose NOT ENOUGH INFO only when the evidence truly cannot support or refute the claim.
- Do not suggest changing the label just to make the answer different.

Do not produce the final answer.

Answer strictly in the following format:
Decision: [KEEP / CHANGE]
Error type: [none / entity_mismatch / negation / numerical_mismatch / temporal_mismatch / causal_direction / increase_decrease_direction / scope_mismatch / insufficient_evidence / wrong_label]
Feedback: ...
""".strip()


def vitaminc_style_refine_prompt(claim, evidence, initial_answer, feedback, iteration=None):
    iteration_info = _iteration_text(iteration)

    return f"""
You are performing contrastive evidence-sensitive self-refinement.

{iteration_info}
Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to decide whether the previous answer should be kept or changed.

Revision policy:
- KEEP the previous label unless the feedback identifies a clear contrastive evidence error.
- CHANGE the label only if a subtle evidence difference clearly changes the verdict.
- If this is not the first refinement iteration, avoid changing the label again unless a new clear evidence-based error is found.
- Do not change the label just to make the answer different.
- Do not change to NOT ENOUGH INFO merely because the evidence is short.
- Choose NOT ENOUGH INFO only when the evidence does not contain enough information to support or refute the claim.
- If the evidence explicitly contradicts the claim, choose REFUTES.
- If the evidence explicitly entails the claim, choose SUPPORTS.

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

Feedback:
{feedback}

Now decide whether to KEEP or CHANGE the previous answer.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Decision: [KEEP / CHANGE]
Explanation: ...
""".strip()
