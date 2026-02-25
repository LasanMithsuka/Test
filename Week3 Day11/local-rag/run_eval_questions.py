from pipeline import answer_question
import json

with open("test_questions.json", encoding="utf-8") as f:
    test_questions = json.load(f)

results = []
for q in test_questions:
    if "question" not in q:
        raise KeyError(f"Missing 'question' field in item id={q.get('id')}. Keys: {list(q.keys())}")

    if "expected_answer" not in q:
        raise KeyError(f"Missing 'expected_answer' field in item id={q.get('id')}. Keys: {list(q.keys())}")

    output = answer_question(q["question"])

    results.append({
        "id": q.get("id"),
        "question": q["question"],
        "expected_answer": q["expected_answer"],
        "actual_answer": output["answer"],
        "retrieved_context": [c["content"] for c in output["retrieval"]["retrieved_chunks"]],
        "model": output.get("model"),
        "retrieval_time_ms": output["retrieval"].get("retrieval_time_ms"),
        "generation_time_ms": output.get("generation_time_ms"),
    })

with open("pipeline_outputs_local.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Generated outputs for {len(results)} questions -> pipeline_outputs_local.json")