import os
import math
from openai import OpenAI

MODEL = os.environ.get(
    "MODEL_PATH",
    "/home1/xuh1010/models/Qwen2.5-7B-Instruct"
)

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)


def call_llm(prompt, temperature=0.0, max_tokens=512):
    """기존과 동일. 기존 방식들(basic/evidence-aware/vitaminc)에서 그대로 사용."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def call_llm_with_logprobs(prompt, temperature=0.0, max_tokens=512, top_logprobs=5):
    """
    logprobs=True로 호출. vLLM은 OpenAI API 호환이라 그대로 지원.

    Returns:
        content       : str   - LLM 생성 텍스트
        label_prob    : float - "Label:" 직후 label 토큰의 확률 (0~1)
                                파싱 실패 시 None
        label_token   : str   - 실제로 생성된 label 토큰 문자열
        raw_logprobs  : list  - 전체 토큰별 logprob 리스트 (디버깅용)

    logprobs 구조 (vLLM / OpenAI 공통):
        response.choices[0].logprobs.content = [
            ChatCompletionTokenLogprob(
                token      = "Label",
                logprob    = -0.01,
                top_logprobs = [
                    TopLogprob(token="Label", logprob=-0.01),
                    TopLogprob(token=" SUPPORTS", logprob=-3.2),
                    ...
                ]
            ),
            ChatCompletionTokenLogprob(token=":", logprob=-0.001, ...),
            ChatCompletionTokenLogprob(token=" SUPPORTS", logprob=-0.05, ...),
            ...
        ]
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    content      = response.choices[0].message.content.strip()
    raw_logprobs = response.choices[0].logprobs.content  # list of ChatCompletionTokenLogprob

    label_prob  = None
    label_token = None

    # "Label:" 다음에 오는 토큰에서 label 확률 추출
    # 토큰 시퀀스를 순회하며 "label" 키워드 위치 탐지
    tokens = [t.token for t in raw_logprobs]

    for i, tok in enumerate(tokens):
        # "Label" 또는 "label" 토큰 탐지
        if tok.strip().lower() == "label" and i + 1 < len(tokens):
            # 바로 다음이 ":" 인지 확인 (또는 ": " 합쳐진 토큰일 수도 있음)
            next_tok = tokens[i + 1].strip()

            # ":" 토큰인 경우 → 그 다음이 label
            if next_tok == ":" and i + 2 < len(tokens):
                label_tok_obj = raw_logprobs[i + 2]
            # ":" 가 label 토큰과 합쳐진 경우 (":SUPPORTS" 등) → 바로 다음
            elif next_tok.startswith(":"):
                label_tok_obj = raw_logprobs[i + 1]
            else:
                continue

            label_token = label_tok_obj.token.strip().strip("[").strip("]")
            label_prob  = math.exp(label_tok_obj.logprob)
            break

    return content, label_prob, label_token, raw_logprobs