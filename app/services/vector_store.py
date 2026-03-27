import json
from pathlib import Path

import faiss
import numpy as np


VECTOR_STORE_DIR = Path("data/vector_store")
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"


def ensure_vector_store_dir() -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    if not embeddings:
        raise ValueError("embeddings must not be empty.")

    embedding_matrix = np.array(embeddings, dtype=np.float32)
    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)

    return index


def save_faiss_index(index: faiss.IndexFlatL2, path: Path = FAISS_INDEX_PATH) -> None:
    ensure_vector_store_dir()
    faiss.write_index(index, str(path))


def load_faiss_index(path: Path = FAISS_INDEX_PATH) -> faiss.IndexFlatL2:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}")

    return faiss.read_index(str(path))


def save_chunks(chunks: list[str], path: Path = CHUNKS_PATH) -> None:
    ensure_vector_store_dir()

    with path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)


def load_chunks(path: Path = CHUNKS_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def search_similar_chunks(
    query_embedding: list[float],
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 3,
) -> list[str]:
    if not query_embedding:
        raise ValueError("query_embedding must not be empty.")

    if not chunks:
        raise ValueError("chunks must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")

    query_vector = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(query_vector, top_k)

    results: list[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])

    return results