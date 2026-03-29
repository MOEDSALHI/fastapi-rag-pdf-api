from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.exceptions import (
    EmbeddingGenerationError,
    LLMResponseGenerationError,
    VectorStoreError,
)
from app.main import app

client = TestClient(app)


def test_ask_returns_answer_and_chunks() -> None:
    with (
        patch(
            "app.api.routes.ask.generate_embeddings",
            return_value=[[0.1, 0.2, 0.3]],
        ),
        patch(
            "app.api.routes.ask.load_faiss_index",
            return_value="fake-index",
        ),
        patch(
            "app.api.routes.ask.load_chunks",
            return_value=["chunk 1", "chunk 2", "chunk 3"],
        ),
        patch(
            "app.api.routes.ask.search_similar_chunks",
            return_value=["chunk 1", "chunk 3"],
        ),
        patch(
            "app.api.routes.ask.generate_rag_answer",
            return_value="Voici la réponse basée sur le document.",
        ),
    ):
        response = client.post(
            "/ask",
            json={
                "question": "Quel est le loyer ?",
                "top_k": 2,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "question": "Quel est le loyer ?",
        "answer": "Voici la réponse basée sur le document.",
        "retrieved_chunks": ["chunk 1", "chunk 3"],
    }


def test_ask_returns_404_when_no_indexed_document_exists() -> None:
    with (
        patch(
            "app.api.routes.ask.generate_embeddings",
            return_value=[[0.1, 0.2, 0.3]],
        ),
        patch(
            "app.api.routes.ask.load_faiss_index",
            side_effect=FileNotFoundError("index not found"),
        ),
    ):
        response = client.post(
            "/ask",
            json={"question": "Quel est le loyer ?"},
        )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No indexed document was found. Upload and index a PDF first."
    }


def test_ask_returns_502_when_embedding_generation_fails() -> None:
    with patch(
        "app.api.routes.ask.generate_embeddings",
        side_effect=EmbeddingGenerationError(
            "An error occurred while generating embeddings."
        ),
    ):
        response = client.post(
            "/ask",
            json={"question": "Quel est le loyer ?"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "An error occurred while generating embeddings."
    }


def test_ask_returns_502_when_llm_generation_fails() -> None:
    with (
        patch(
            "app.api.routes.ask.generate_embeddings",
            return_value=[[0.1, 0.2, 0.3]],
        ),
        patch(
            "app.api.routes.ask.load_faiss_index",
            return_value="fake-index",
        ),
        patch(
            "app.api.routes.ask.load_chunks",
            return_value=["chunk 1", "chunk 2"],
        ),
        patch(
            "app.api.routes.ask.search_similar_chunks",
            return_value=["chunk 2"],
        ),
        patch(
            "app.api.routes.ask.generate_rag_answer",
            side_effect=LLMResponseGenerationError(
                "An error occurred while generating the answer."
            ),
        ),
    ):
        response = client.post(
            "/ask",
            json={"question": "Quel est le loyer ?"},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "An error occurred while generating the answer."
    }


def test_ask_returns_422_for_empty_question() -> None:
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_returns_500_when_no_relevant_chunks_are_found() -> None:
    with (
        patch(
            "app.api.routes.ask.generate_embeddings",
            return_value=[[0.1, 0.2, 0.3]],
        ),
        patch(
            "app.api.routes.ask.load_faiss_index",
            return_value="fake-index",
        ),
        patch(
            "app.api.routes.ask.load_chunks",
            return_value=["chunk 1", "chunk 2"],
        ),
        patch(
            "app.api.routes.ask.search_similar_chunks",
            return_value=[],
        ),
    ):
        response = client.post(
            "/ask",
            json={"question": "Quel est le loyer ?"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "No relevant chunks were found for the given question."
    }


def test_ask_returns_500_when_vector_store_loading_fails() -> None:
    with (
        patch(
            "app.api.routes.ask.generate_embeddings",
            return_value=[[0.1, 0.2, 0.3]],
        ),
        patch(
            "app.api.routes.ask.load_faiss_index",
            side_effect=VectorStoreError(
                "An error occurred while loading the FAISS index from data/vector_store/index.faiss."
            ),
        ),
    ):
        response = client.post(
            "/ask",
            json={"question": "Quel est le loyer ?"},
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An error occurred while loading the FAISS index from data/vector_store/index.faiss."
    }
