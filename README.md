# LLM RAG Assistant Prototype

A lightweight Retrieval-Augmented Generation (RAG) assistant that answers questions using your own documents.

## What it does
- Ingests `.txt` / `.md` documents from `data/`
- Chunks documents with overlap
- Embeds chunks using SentenceTransformers
- Indexes embeddings with FAISS
- Retrieves top-k relevant chunks for each query
- Generates grounded answers using OpenAI (optional)
- Tracks latency and basic retrieval stats

If no `OPENAI_API_KEY` is provided, the app still runs and returns a grounded extractive snippet from the top retrieved chunk.

## Quickstart

### 1) Install
```bash
pip install -r requirements.txt