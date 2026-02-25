import os
import json
import time
from openai import RateLimitError

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.models import AzureOpenAIModel


def _normalize_azure_endpoint(endpoint: str) -> str:
    
    
    endpoint = (endpoint or "").strip().rstrip("/")
    if "/openai" in endpoint:
        endpoint = endpoint.split("/openai")[0]
    return endpoint


def main():
    with open("pipeline_outputs_cloud.json", encoding="utf-8") as f:
        results = json.load(f)

    azure_endpoint = _normalize_azure_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT"))
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("OPENAI_API_VERSION") or "2025-01-01-preview"
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_DEPLOYMENT_NAME") or "gpt-4.1-mini"
    model_name = os.getenv("AZURE_MODEL_NAME") or "gpt-4.1-mini"

    if not azure_endpoint or not api_key:
        raise RuntimeError(
            "Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY in your environment."
        )

    judge = AzureOpenAIModel(
        model=model_name,
        deployment_name=deployment,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    test_cases = []
    for r in results:
        test_cases.append(
            LLMTestCase(
                input=r["question"],
                actual_output=r["actual_answer"],
                expected_output=r["expected_answer"],
                retrieval_context=r.get("retrieved_context", []),
            )
        )

    metrics = [
        FaithfulnessMetric(threshold=0.7, include_reason=True, model=judge),
        AnswerRelevancyMetric(threshold=0.7, include_reason=True, model=judge),
        ContextualRelevancyMetric(threshold=0.7, include_reason=True, model=judge),
        ContextualRecallMetric(threshold=0.7, include_reason=True, model=judge),
        ContextualPrecisionMetric(threshold=0.7, include_reason=True, model=judge),
    ]

    ALL_RESULTS = []

    for i, tc in enumerate(test_cases, start=1):
        print(f"Evaluating CLOUD test case {i}/{len(test_cases)} ")

        while True:
            try:
                one = evaluate([tc], metrics)
                if isinstance(one, tuple):
                    one = one[0]
                ALL_RESULTS.extend(one)
                break
            except RateLimitError:
                wait_seconds = 30
                print(f"[Rate limit] Waiting {wait_seconds}s then retrying...")
                time.sleep(wait_seconds)

        time.sleep(2)

    serializable = []
    for r in ALL_RESULTS:
        serializable.append({
            "name": getattr(r, "name", None),
            "success": getattr(r, "success", None),
            "input": getattr(getattr(r, "test_case", None), "input", None),
            "actual_output": getattr(getattr(r, "test_case", None), "actual_output", None),
            "expected_output": getattr(getattr(r, "test_case", None), "expected_output", None),
            "retrieval_context": getattr(getattr(r, "test_case", None), "retrieval_context", None),
            "metrics": [
                {
                    "metric": getattr(m, "metric", None) or getattr(m, "name", None),
                    "score": getattr(m, "score", None),
                    "threshold": getattr(m, "threshold", None),
                    "success": getattr(m, "success", None),
                    "reason": getattr(m, "reason", None),
                }
                for m in (getattr(r, "metrics_data", None) or getattr(r, "metrics", None) or [])
            ],
        })

    with open("eval_results_cloud.json", "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print("Done. Saved -> eval_results_cloud.json")


if __name__ == "__main__":
    main()