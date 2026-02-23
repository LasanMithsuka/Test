import json
from retrieve import retrieve_chunks
from generate import generate_answer

def answer_question(query: str) -> dict:
    retrieval = retrieve_chunks(query)
    generation = generate_answer(query=query, chunks=retrieval["retrieved_chunks"])
    return {**generation, "retrieval": retrieval}

if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or type 'exit'): ").strip()
        if q.lower() in ["exit", "quit"]:
            break

        result = answer_question(q)
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(result, indent=2))