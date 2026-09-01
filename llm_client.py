"""OpenAI-compatible client used by the baseline experiments.

The default configuration targets a local vLLM server. Override the model,
endpoint, or API key with environment variables when needed.
"""

import os

from openai import OpenAI

MODEL = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def call_llm(prompt, temperature=0.0, max_tokens=512):
    """Generate one response from the configured OpenAI-compatible server."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()
