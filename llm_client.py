import os
from openai import OpenAI

# 실행할 때 MODEL_PATH 환경변수로 모델 경로를 넘길 수 있음.
# 예:
# MODEL_PATH=/home1/xuh1010/models/Qwen2.5-7B-Instruct python run_experiment.py
MODEL = os.environ.get(
    "MODEL_PATH",
    "/home1/xuh1010/models/Qwen2.5-7B-Instruct"
)

client = OpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="EMPTY",
)


def call_llm(prompt, temperature=0.0, max_tokens=512):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content.strip()