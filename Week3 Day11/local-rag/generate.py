import time
from openai import OpenAI

from config import (
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
    SYSTEM_PROMPT, TEMPERATURE, MAX_TOKENS
)

def generate_answer(query: str, chunks: list[dict]) -> dict:
    start = time.time()

    context = "\n\n---\n\n".join([c["content"] for c in chunks])

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    resp = client.chat.completions.create(
        model=LLM_MODEL,
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
        "model": LLM_MODEL,
        "generation_time_ms": int((time.time() - start) * 1000),
        "tokens_generated": tokens_generated,
    }