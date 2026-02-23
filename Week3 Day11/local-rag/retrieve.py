import time
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

from config import DB_PATH, EMBEDDING_MODEL, TOP_K

def retrieve_chunks(query: str) -> dict:
    start = time.time()

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    db = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    results = db.similarity_search_with_relevance_scores(query, k=TOP_K)

    retrieved = []
    for doc, score in results:
        meta = doc.metadata or {}
        retrieved.append({
            "content": doc.page_content,
            "source": meta.get("source"),
            "page": meta.get("page"),
            "relevance_score": float(score),
        })

    return {
        "query": query,
        "retrieved_chunks": retrieved,
        "retrieval_time_ms": int((time.time() - start) * 1000),
    }