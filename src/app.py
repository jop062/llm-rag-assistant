import os
from dotenv import load_dotenv

from ingest import ingest_directory
from embed import build_embeddings
from retrieve import build_faiss_index, Retriever
from generate import generate_answer
from evaluate import retrieval_stats, simple_grounding_check
from utils import ensure_dir


def build_pipeline():
    load_dotenv()

    data_dir = "data"
    index_dir = "index"
    ensure_dir(index_dir)

    chunk_size = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
    embed_model = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = int(os.getenv("TOP_K", "5"))

    chunks_path = ingest_directory(data_dir=data_dir, out_dir=index_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    vectors_path, metadata_path = build_embeddings(chunks_path=chunks_path, out_dir=index_dir, embed_model_name=embed_model)
    index_path = build_faiss_index(vectors_path=vectors_path, out_dir=index_dir)

    retriever = Retriever(embed_model_name=embed_model, index_path=index_path, metadata_path=metadata_path)
    return retriever, top_k


def main():
    print("\nLLM RAG Assistant (local embeddings + FAISS)")
    print("Type a question. Type 'exit' to quit.\n")

    retriever, top_k = build_pipeline()

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        retrieved = retriever.search(q, top_k=top_k)
        gen = generate_answer(q, retrieved)

        rstats = retrieval_stats(retrieved)
        gcheck = simple_grounding_check(gen["answer"], retrieved)

        print("\nAssistant:\n" + gen["answer"])
        print("\n---")
        print(f"Used LLM: {gen['used_llm']}")
        print(f"Latency: {gen['latency_ms']:.1f} ms")
        print(f"Top score: {rstats['top_score']:.3f} | Avg score: {rstats['avg_score']:.3f}")
        print(f"Retrieved docs: {', '.join(rstats.get('doc_ids', []))}")
        if gen["used_llm"]:
            print(f"Docs mentioned in answer: {gcheck['mentioned_count']} -> {gcheck['docs_mentioned']}")
        print("---\n")


if __name__ == "__main__":
    main()