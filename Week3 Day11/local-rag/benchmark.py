import json
import time
from openai import OpenAI
from config import LLM_BASE_URL, LLM_API_KEY

MODELS = ["qwen3:1.7b", "qwen3:4b"]  # make sure you have both pulled in Ollama

PROMPTS = [
    {"category": "factual", "prompt": "What causes tides on Earth? Answer in 2-3 sentences."},
    {"category": "reasoning", "prompt": "If a train travels 120km in 1.5 hours and then 80km in 1 hour, what is the average speed?"},
    {"category": "summarization", "prompt": "Summarize in 2 sentences: Large language models can draft and summarize but may hallucinate facts."},
    {"category": "structured", "prompt": 'List 3 European capitals. Respond ONLY with valid JSON: [{"city":"...","country":"..."}]'},
    {"category": "code", "prompt": "Write a Python function that checks if a string is a palindrome."},
    {"category": "instruction", "prompt": "Respond with exactly 3 bullet points about cybersecurity best practices."},
]

def run_one(model: str, prompt: str) -> dict:
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    start = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400
    )
    elapsed = time.time() - start

    text = resp.choices[0].message.content
    usage = resp.usage

    completion_tokens = usage.completion_tokens if usage else None
    tps = round(completion_tokens / elapsed, 2) if completion_tokens else None

    return {
        "model": model,
        "category": None,
        "prompt": prompt,
        "response": text,
        "time_seconds": round(elapsed, 2),
        "completion_tokens": completion_tokens,
        "tokens_per_second": tps,
        "subjective_quality_1_to_5": None
    }

def main():
    results = []
    for model in MODELS:
        for p in PROMPTS:
            print(f"Running {model} | {p['category']} ...")
            r = run_one(model, p["prompt"])
            r["category"] = p["category"]
            results.append(r)

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved benchmark_results.json")

if __name__ == "__main__":
    main()