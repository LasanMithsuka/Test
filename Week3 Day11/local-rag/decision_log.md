# Decision Log — Local RAG Pipeline

## Model Selection
Chosen LLM: qwen3:1.7b

Why:
Fast on local machine and produces good answers in tests.

Other models tested:
qwen3:4b
phi4-mini

## Embedding Model
Chosen: nomic-embed-text

Reason:
Good semantic search and works well with ChromaDB.

## Chunking Strategy
Chunk size: 600  
Overlap: 120

Reason:
Improves retrieval context without losing information.

## Retrieval
Top-K: 4

Reason:
Balanced between context quality and speed.

## Observations

Works well:
Password policy questions answered correctly.

Failures:
Complex reasoning questions sometimes vague.

## If I had more time

- Try hybrid search
- Test larger models
- Improve evaluation metrics