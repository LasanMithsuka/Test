# Day 11 Capstone — Local Document Q&A (RAG)

## Requirements
- Windows 11
- Python 3.11
- Ollama running locally

## Models (Ollama)
Pull required models:
- `ollama pull qwen3:1.7b`
- `ollama pull qwen3:4b`
- `ollama pull nomic-embed-text`

## Setup
Create venv and install deps:
- `python -m venv .venv`
- Activate venv
- `pip install -U openai chromadb langchain langchain-community langchain-core langchain-text-splitters langchain-ollama pypdf`

## Run Ingestion
- `python ingest.py`

## Run Full Pipeline
- `python pipeline.py`

## Benchmark
- `python benchmark.py`
Outputs: `benchmark_results.json`

## Test Questions
See `test_questions.json` for 15 Q&A pairs used for evaluation on Day 12.

## Notes
- `chroma_db/` and `.venv/` are gitignored.