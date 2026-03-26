from openai import OpenAI

from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationError


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=settings.openai_timeout,
    )


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.

    Args:
        texts: List of text chunks.

    Returns:
        List of embedding vectors.
    """
    cleaned_texts = [text.strip() for text in texts if text and text.strip()]

    if not cleaned_texts:
        return []

    try:
        client = get_openai_client()
        response = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=cleaned_texts,
        )

        return [item.embedding for item in response.data]

    except Exception as exc:
        raise EmbeddingGenerationError(
            "An error occurred while generating embeddings."
        ) from exc