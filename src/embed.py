import os
import json
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer
from utils import ensure_dir


def load_chunks(chunks_path: str) -> List[Dict]:
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_embeddings(
    chunks_path: str,
    out_dir: str,
    embed_model_name: str
) -> Tuple[str, str]:
    """
    Creates embeddings for each chunk and writes:
      - vectors.npy
      - metadata.jsonl
    Returns (vectors_path, metadata_path)
    """
    ensure_dir(out_dir)
    vectors_path = os.path.join(out_dir, "vectors.npy")
    metadata_path = os.path.join(out_dir, "metadata.jsonl")

    chunks = load_chunks(chunks_path)
    texts = [c["text"] for c in chunks]

    print(f"[embed] Loading model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    print(f"[embed] Embedding {len(texts)} chunks...")
    vectors = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype("float32")

    np.save(vectors_path, vectors)

    with open(metadata_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"[embed] Saved vectors -> {vectors_path}")
    print(f"[embed] Saved metadata -> {metadata_path}")
    return vectors_path, metadata_path