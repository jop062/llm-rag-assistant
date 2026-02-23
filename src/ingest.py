import os
import json
from typing import List
from utils import Chunk, read_text_file, chunk_text, list_data_files, ensure_dir


def ingest_directory(data_dir: str, out_dir: str, chunk_size: int, chunk_overlap: int) -> str:
    """
    Reads .txt/.md files from data_dir, chunks them, and writes chunks.jsonl.
    Returns path to chunks.jsonl.
    """
    ensure_dir(out_dir)
    out_path = os.path.join(out_dir, "chunks.jsonl")

    files = list_data_files(data_dir)
    if not files:
        raise FileNotFoundError(f"No .txt or .md files found in: {data_dir}")

    total_chunks = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for file_path in files:
            doc_id = os.path.basename(file_path)
            raw = read_text_file(file_path)
            parts = chunk_text(raw, chunk_size=chunk_size, overlap=chunk_overlap)

            for i, text in enumerate(parts):
                c = Chunk(doc_id=doc_id, chunk_id=i, text=text)
                f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[ingest] Wrote {total_chunks} chunks -> {out_path}")
    return out_path