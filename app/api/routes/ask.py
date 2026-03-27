from fastapi import APIRouter, HTTPException, status

from app.core.exceptions import (
    EmbeddingGenerationError,
    LLMResponseGenerationError,
)
from app.schemas.ask import AskRequest, AskResponse
from app.services.embeddings import generate_embeddings
from app.services.llm import generate_rag_answer
from app.services.vector_store import (
    load_chunks,
    load_faiss_index,
    search_similar_chunks,
)

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    status_code=status.HTTP_200_OK,
)
def ask_question(payload: AskRequest) -> AskResponse:
    try:
        query_embeddings = generate_embeddings([payload.question])
        if not query_embeddings:
            raise EmbeddingGenerationError("Failed to generate question embedding.")

        query_embedding = query_embeddings[0]

        index = load_faiss_index()
        chunks = load_chunks()

        retrieved_chunks = search_similar_chunks(
            query_embedding=query_embedding,
            index=index,
            chunks=chunks,
            top_k=payload.top_k,
        )

        answer = generate_rag_answer(
            question=payload.question,
            context_chunks=retrieved_chunks,
        )

        return AskResponse(
            question=payload.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No indexed document was found. Upload and index a PDF first."
            ),
        ) from exc
    except EmbeddingGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except LLMResponseGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc