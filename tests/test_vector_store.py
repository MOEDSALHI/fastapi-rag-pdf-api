from pathlib import Path

import pytest

from app.services.vector_store import (
    build_faiss_index,
    load_chunks,
    load_faiss_index,
    save_chunks,
    save_faiss_index,
    search_similar_chunks,
)


def test_build_faiss_index_raises_error_for_empty_embeddings() -> None:
    with pytest.raises(ValueError, match="embeddings must not be empty."):
        build_faiss_index([])


def test_build_and_search_similar_chunks_returns_expected_results() -> None:
    chunks = [
        "Le loyer mensuel est de 950 euros.",
        "La durée du bail est de 3 ans.",
        "Le dépôt de garantie est de deux mois.",
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],
    ]

    index = build_faiss_index(embeddings)

    query_embedding = [1.0, 0.0]
    results = search_similar_chunks(
        query_embedding=query_embedding,
        index=index,
        chunks=chunks,
        top_k=2,
    )

    assert len(results) == 2
    assert "Le loyer mensuel est de 950 euros." in results
    assert "Le dépôt de garantie est de deux mois." in results


def test_search_similar_chunks_raises_error_for_empty_query_embedding() -> None:
    index = build_faiss_index([[1.0, 0.0]])
    chunks = ["Un chunk"]

    with pytest.raises(ValueError, match="query_embedding must not be empty."):
        search_similar_chunks([], index, chunks)


def test_search_similar_chunks_raises_error_for_empty_chunks() -> None:
    index = build_faiss_index([[1.0, 0.0]])

    with pytest.raises(ValueError, match="chunks must not be empty."):
        search_similar_chunks([1.0, 0.0], index, [])


def test_search_similar_chunks_raises_error_for_invalid_top_k() -> None:
    index = build_faiss_index([[1.0, 0.0]])
    chunks = ["Un chunk"]

    with pytest.raises(ValueError, match="top_k must be greater than 0."):
        search_similar_chunks([1.0, 0.0], index, chunks, top_k=0)


def test_save_and_load_chunks(tmp_path: Path) -> None:
    chunks = [
        "Premier chunk",
        "Deuxième chunk",
    ]
    file_path = tmp_path / "chunks.json"

    save_chunks(chunks, path=file_path)
    loaded_chunks = load_chunks(path=file_path)

    assert loaded_chunks == chunks


def test_save_and_load_faiss_index(tmp_path: Path) -> None:
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    index = build_faiss_index(embeddings)
    file_path = tmp_path / "index.faiss"

    save_faiss_index(index, path=file_path)
    loaded_index = load_faiss_index(path=file_path)

    assert loaded_index.ntotal == 2