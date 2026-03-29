import json
from pathlib import Path

import faiss
import numpy as np

from app.core.exceptions import VectorStoreError


VECTOR_STORE_DIR = Path("data/vector_store")
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.json"


def ensure_vector_store_dir() -> None:
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)


def build_faiss_index(embeddings: list[list[float]]) -> faiss.IndexFlatL2:
    if not embeddings:
        raise ValueError("embeddings must not be empty.")

    first_dimension = len(embeddings[0])
    if first_dimension == 0:
        raise ValueError("embedding vectors must not be empty.")

    if any(len(vector) != first_dimension for vector in embeddings):
        raise ValueError("all embedding vectors must have the same dimension.")

    try:
        embedding_matrix = np.array(embeddings, dtype=np.float32)
        dimension = embedding_matrix.shape[1]

        index = faiss.IndexFlatL2(dimension)
        index.add(embedding_matrix)

        return index
    except Exception as exc:
        raise VectorStoreError(
            "An error occurred while building the FAISS index."
        ) from exc


def save_faiss_index(index: faiss.IndexFlatL2, path: Path = FAISS_INDEX_PATH) -> None:
    try:
        ensure_vector_store_dir()
        faiss.write_index(index, str(path))
    except Exception as exc:
        raise VectorStoreError(
            f"An error occurred while saving the FAISS index to {path}."
        ) from exc


def load_faiss_index(path: Path = FAISS_INDEX_PATH) -> faiss.IndexFlatL2:
    if not path.exists():
        raise FileNotFoundError(f"FAISS index not found at: {path}")

    try:
        return faiss.read_index(str(path))
    except Exception as exc:
        raise VectorStoreError(
            f"An error occurred while loading the FAISS index from {path}."
        ) from exc


def save_chunks(chunks: list[str], path: Path = CHUNKS_PATH) -> None:
    try:
        ensure_vector_store_dir()
        with path.open("w", encoding="utf-8") as file:
            json.dump(chunks, file, ensure_ascii=False, indent=2)
    except Exception as exc:
        raise VectorStoreError(
            f"An error occurred while saving chunks to {path}."
        ) from exc


def load_chunks(path: Path = CHUNKS_PATH) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Chunks file not found at: {path}")

    try:
        with path.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not isinstance(chunks, list):
            raise VectorStoreError("Chunks file content must be a list.")

        return chunks
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"An error occurred while loading chunks from {path}."
        ) from exc


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

    try:
        query_vector = np.array([query_embedding], dtype=np.float32)
        _, indices = index.search(query_vector, top_k)

        results: list[str] = []
        for idx in indices[0]:
            if 0 <= idx < len(chunks):
                results.append(chunks[idx])

        return results
    except Exception as exc:
        raise VectorStoreError(
            "An error occurred while searching similar chunks."
        ) from exc
