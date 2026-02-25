import json
import time
import re
from typing import Any, Dict, List

from openai import RateLimitError

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models.llms.azure_model import AzureOpenAIModel


INPUT_FILE = "pipeline_outputs_local.json"
OUTPUT_FILE = "deepeval_results.json"

def clean_retrieval_context(chunks):
    
    if not chunks:
        return []

    bad_line_patterns = [
        r"price point",
        r"Target implementation date",
        r"before a final price point is set",
        r"Next leadership team meeting",
        r"reviewed and approved by",
    ]

    cleaned_chunks = []
    for chunk in chunks:
        lines = chunk.splitlines()
        kept = []
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if any(re.search(pat, l, re.IGNORECASE) for pat in bad_line_patterns):
                continue
            kept.append(line)
        if kept:
            cleaned_chunks.append("\n".join(kept))

    return cleaned_chunks

def to_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def flatten_eval_return(one_result) -> List[Any]:
   
    if isinstance(one_result, tuple) and len(one_result) >= 1:
        return to_list(one_result[0])

    tr = getattr(one_result, "test_results", None)
    if tr is not None:
        return to_list(tr)

    return to_list(one_result)


def extract_metrics(result_obj):
    return (
        getattr(result_obj, "metrics_data", None)
        or getattr(result_obj, "metrics", None)
        or []
    )


def serialize_metric(m) -> Dict[str, Any]:
    return {
        "metric": getattr(m, "metric", None) or getattr(m, "name", None),
        "score": getattr(m, "score", None),
        "threshold": getattr(m, "threshold", None),
        "success": getattr(m, "success", None),
        "reason": getattr(m, "reason", None),
        "error": getattr(m, "error", None),
    }


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        rows = json.load(f)

    judge_model = AzureOpenAIModel() 

    metrics = [
        FaithfulnessMetric(model=judge_model, async_mode=False),
        AnswerRelevancyMetric(model=judge_model, async_mode=False),
        ContextualRelevancyMetric(model=judge_model, async_mode=False),
        ContextualRecallMetric(model=judge_model, async_mode=False),
        ContextualPrecisionMetric(model=judge_model, async_mode=False),
    ]

    serializable: List[Dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        
        raw_ctx = row.get("retrieved_context") or []
        filtered_ctx = clean_retrieval_context(raw_ctx)
        tc = LLMTestCase(
            input=row.get("question", "") or "",
            actual_output=row.get("actual_answer", "") or "",
            expected_output=row.get("expected_answer", "") or "",
            retrieval_context=filtered_ctx,
        
        )

        print(f"\n--- Evaluating test case {i}/{len(rows)} ---")

        backoff = 10
        while True:
            try:
                one = evaluate([tc], metrics) 
                results = flatten_eval_return(one)

                r = results[0] if results else None
                metrics_out = [serialize_metric(m) for m in extract_metrics(r)] if r else []

                serializable.append({
                    "id": row.get("id"),
                    "model": row.get("model"),
                    "input": tc.input,
                    "actual_output": tc.actual_output,
                    "expected_output": tc.expected_output,
                    "retrieval_context": tc.retrieval_context,
                    "success": getattr(r, "success", None) if r else None,
                    "name": getattr(r, "name", None) if r else None,
                    "metrics": metrics_out,
                    "retrieval_time_ms": row.get("retrieval_time_ms"),
                    "generation_time_ms": row.get("generation_time_ms"),
                })
                break

            except RateLimitError:
                print(f"[Rate limit] Waiting {backoff}s then retrying...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)

            except Exception as e:
                msg = str(e)
                if "RateLimitReached" in msg or "429" in msg:
                    print(f"[Rate limit/wrapped] Waiting {backoff}s then retrying...")
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                raise

        time.sleep(3)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f" Done. Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()