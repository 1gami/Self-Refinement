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
# Basic Self-Refine: Conservative version
# ─────────────────────────────────────────────────────────────

def basic_feedback_prompt(claim, evidence, initial_answer):
    return f"""
You are providing self-feedback for a fact verification answer.

Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Initial answer:
{initial_answer}

Your task is to critically review the initial answer.

Check whether:
1. The predicted label is correct based on the evidence.
2. The explanation is faithful to the evidence.
3. The answer relies on unsupported assumptions or outside knowledge.
4. The evidence actually supports, refutes, or is insufficient for the claim.
5. A different label would be better justified by the evidence.

Do not produce the final revised answer yet.
Provide constructive feedback that can help improve the answer.
If the initial answer is already correct, explain why no revision is needed.
If the initial answer is wrong or weakly justified, explain what should be revised and why.

Answer strictly in the following format:
Feedback:
- Label assessment: ...
- Evidence assessment: ...
- Potential issue: ...
- Suggested revision: ...
""".strip()

def basic_refine_prompt(claim, evidence, initial_answer, feedback):
    return f"""
You are refining a fact verification answer using self-feedback.

Use only the provided evidence. Do not use outside knowledge.

Claim:
{claim}

Evidence:
{evidence}

Initial answer:
{initial_answer}

Self-feedback:
{feedback}

Your task is to produce an improved final answer.

You may keep the initial label if it is well supported by the evidence.
You should revise the label if the feedback shows that another label is better supported.
The final label must be one of:
- SUPPORTS: The evidence clearly entails that the claim is true.
- REFUTES: The evidence clearly contradicts the claim.
- NOT ENOUGH INFO: The evidence does not provide enough information to determine whether the claim is true or false.

Base the final answer only on the evidence and the self-feedback.
Do not simply repeat the initial answer unless it is justified.

Answer strictly in the following format:
Label: [SUPPORTS / REFUTES / NOT ENOUGH INFO]
Explanation: ...
""".strip()


# ─────────────────────────────────────────────────────────────
# Evidence-aware Self-Refine: Conservative version
# ─────────────────────────────────────────────────────────────

def evidence_aware_feedback_prompt(claim, evidence, initial_answer):
    return f"""
You are a careful fact verification reviewer.

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
- Do not recommend NOT ENOUGH INFO merely because the evidence is short.

Do not produce the final answer.

Answer strictly in the following format:
Decision: [KEEP / CHANGE]
Error type: [none / entity_mismatch / directionality_error / numerical_mismatch / temporal_mismatch / negation_missed / overclaiming / insufficient_evidence / wrong_label]
Feedback: ...
""".strip()


def evidence_aware_refine_prompt(claim, evidence, initial_answer, feedback):
    return f"""
You are performing evidence-aware self-refinement for fact verification.

Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to verify whether the previous answer should be kept or changed.

Revision policy:
- KEEP the previous label unless the feedback identifies a clear evidence-based error.
- CHANGE the label only when the claim-evidence relationship clearly supports a different label.
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
# VitaminC-style Self-Refine: Conservative version
# ─────────────────────────────────────────────────────────────

def vitaminc_style_feedback_prompt(claim, evidence, initial_answer):
    return f"""
You are a fact verification reviewer trained to detect subtle evidence changes.

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
- Do not change to NOT ENOUGH INFO merely because the evidence is short or indirect.
- Choose NOT ENOUGH INFO only when the evidence truly cannot support or refute the claim.

Do not produce the final answer.

Answer strictly in the following format:
Decision: [KEEP / CHANGE]
Error type: [none / entity_mismatch / negation / numerical_mismatch / temporal_mismatch / causal_direction / increase_decrease_direction / scope_mismatch / insufficient_evidence / wrong_label]
Feedback: ...
""".strip()


def vitaminc_style_refine_prompt(claim, evidence, initial_answer, feedback):
    return f"""
You are performing contrastive evidence-sensitive self-refinement.

Use only the provided evidence. Do not use outside knowledge.

Your task is NOT to generate a new answer from scratch.
Your task is to decide whether the previous answer should be kept or changed.

Revision policy:
- KEEP the previous label unless the feedback identifies a clear contrastive evidence error.
- CHANGE the label only if a subtle evidence difference clearly changes the verdict.
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