import os
import time
import requests
from typing import List, Dict


SYSTEM_PROMPT = """You are a helpful assistant.
Answer using ONLY the provided context.
If the answer is not in the context, say: "I don't know based on the provided documents."
Keep answers concise and cite which doc you used (doc_id)."""


def format_context(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"[doc={c['doc_id']} chunk={c['chunk_id']} score={c['score']:.3f}]\n{c['text']}\n")
    return "\n---\n".join(lines)


def extractive_fallback_answer(question: str, chunks: List[Dict]) -> str:
    """
    If no API key, return a simple extractive response from top chunk(s).
    Not as smart, but still 'works' and stays grounded.
    """
    if not chunks:
        return "I don't know based on the provided documents."

    top = chunks[0]
    snippet = top["text"].strip()
    # keep it short
    snippet = snippet[:500] + ("..." if len(snippet) > 500 else "")
    return (
        "No LLM key detected, so here's a grounded snippet from the most relevant document:\n\n"
        f"Source: {top['doc_id']} (chunk {top['chunk_id']})\n\n"
        f"{snippet}"
    )


def openai_chat_completion(model: str, api_key: str, question: str, context: str) -> str:
    """
    Uses OpenAI's Responses-style endpoint via HTTPS (no SDK required).
    This keeps the project simple and portable.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"},
        ],
        "temperature": 0.2,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_answer(question: str, retrieved: List[Dict]) -> Dict:
    """
    Returns dict with:
      - answer
      - used_llm (bool)
      - latency_ms
    """
    context = format_context(retrieved)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

    start = time.perf_counter()
    if api_key:
        answer = openai_chat_completion(model=model, api_key=api_key, question=question, context=context)
        used_llm = True
    else:
        answer = extractive_fallback_answer(question, retrieved)
        used_llm = False
    latency_ms = (time.perf_counter() - start) * 1000.0

    return {"answer": answer, "used_llm": used_llm, "latency_ms": latency_ms}