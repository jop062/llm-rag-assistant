import os
import re
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    """
    Simple character-based chunking with overlap.
    Good enough for portfolio projects; easy to understand + stable.
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


def list_data_files(data_dir: str) -> List[str]:
    paths = []
    for name in os.listdir(data_dir):
        if name.lower().endswith((".txt", ".md")):
            paths.append(os.path.join(data_dir, name))
    return sorted(paths)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)