import os
import time
from openai import OpenAI, AzureOpenAI

from config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
)

def _is_azure() -> bool:
    return bool(os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY"))

def generate_answer(query: str, chunks: list[dict]) -> dict:
    start = time.time()

    context = "\n\n---\n\n".join([c["content"] for c in chunks])

    if _is_azure():
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("OPENAI_API_VERSION", "2025-01-01-preview"),
        )
        model_name = os.getenv("AZURE_OPENAI_DEPLOYMENT") or LLM_MODEL
    else:
        client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
        model_name = LLM_MODEL

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    answer = resp.choices[0].message.content

    tokens_generated = None
    if resp.usage and resp.usage.completion_tokens is not None:
        tokens_generated = int(resp.usage.completion_tokens)

    return {
        "query": query,
        "answer": answer,
        "context_used": context,
        "model": model_name,
        "generation_time_ms": int((time.time() - start) * 1000),
        "tokens_generated": tokens_generated,
    }