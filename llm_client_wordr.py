"""OpenAI-compatible client with token log-probability support.

WordR/WordCand use log probabilities from a local vLLM server to measure
confidence changes after perturbing candidate rationale spans.
"""

import math
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


def call_llm_with_logprobs(prompt, temperature=0.0, max_tokens=512, top_logprobs=5):
    """Generate a response and estimate the probability of the emitted label.

    Returns:
        content: Generated text.
        label_prob: Probability of the first label token following ``Label:``.
        label_token: The emitted label token, when found.
        raw_logprobs: Token-level log-probability objects returned by the server.
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=top_logprobs,
    )

    content = response.choices[0].message.content.strip()
    raw_logprobs = response.choices[0].logprobs.content

    label_prob = None
    label_token = None
    tokens = [token_logprob.token for token_logprob in raw_logprobs]

    for i, token in enumerate(tokens):
        if token.strip().lower() != "label" or i + 1 >= len(tokens):
            continue

        next_token = tokens[i + 1].strip()
        if next_token == ":" and i + 2 < len(tokens):
            label_token_obj = raw_logprobs[i + 2]
        elif next_token.startswith(":"):
            label_token_obj = raw_logprobs[i + 1]
        else:
            continue

        label_token = label_token_obj.token.strip().strip("[").strip("]")
        label_prob = math.exp(label_token_obj.logprob)
        break

    return content, label_prob, label_token, raw_logprobs
