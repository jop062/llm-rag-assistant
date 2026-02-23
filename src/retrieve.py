import os
import json
import numpy as np
import faiss
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from utils import ensure_dir


def load_metadata(metadata_path: str) -> List[Dict]:
    items = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items


def build_faiss_index(vectors_path: str, out_dir: str) -> str:
    """
    Builds and saves a FAISS index from vectors.npy
    Returns path to index file.
    """
    ensure_dir(out_dir)
    index_path = os.path.join(out_dir, "faiss.index")

    vectors = np.load(vectors_path).astype("float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)  # Inner Product; vectors are normalized => cosine similarity
    index.add(vectors)

    faiss.write_index(index, index_path)
    print(f"[index] FAISS index saved -> {index_path} (n={index.ntotal}, dim={dim})")
    return index_path


class Retriever:
    def __init__(self, embed_model_name: str, index_path: str, metadata_path: str):
        self.model = SentenceTransformer(embed_model_name)
        self.index = faiss.read_index(index_path)
        self.metadata = load_metadata(metadata_path)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        qvec = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qvec, top_k)

        results = []
        for score, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i == -1:
                continue
            item = dict(self.metadata[i])
            item["score"] = float(score)
            results.append(item)
        return results