import pytest
from unittest.mock import Mock, patch

from app.core.exceptions import EmbeddingGenerationError
from app.services.embeddings import generate_embeddings


def test_generate_embeddings_returns_empty_list_for_empty_input() -> None:
    assert generate_embeddings([]) == []


def test_generate_embeddings_filters_empty_strings() -> None:
    with patch("app.services.embeddings.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_embeddings(["   ", "valid chunk", ""])

        assert result == [[0.1, 0.2, 0.3]]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["valid chunk"],
        )


def test_generate_embeddings_returns_vectors() -> None:
    with patch("app.services.embeddings.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2]),
            Mock(embedding=[0.3, 0.4]),
        ]
        mock_client.embeddings.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = generate_embeddings(["chunk 1", "chunk 2"])

        assert result == [
            [0.1, 0.2],
            [0.3, 0.4],
        ]
        mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["chunk 1", "chunk 2"],
        )


def test_generate_embeddings_raises_embedding_generation_error() -> None:
    with patch("app.services.embeddings.get_openai_client") as mock_get_client:
        mock_client = Mock()
        mock_client.embeddings.create.side_effect = Exception("OpenAI error")
        mock_get_client.return_value = mock_client

        with pytest.raises(
            EmbeddingGenerationError,
            match="An error occurred while generating embeddings.",
        ):
            generate_embeddings(["chunk 1"])
