from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.exceptions import (
    DocumentProcessingError,
    EmbeddingGenerationError,
    InvalidFileTypeError,
    PDFExtractionError,
    VectorStoreError,
)
from app.schemas.upload import UploadAndIndexResponse
from app.services.chunking import chunk_text
from app.services.embeddings import generate_embeddings
from app.services.pdf_extractor import extract_text_from_pdf_bytes
from app.services.vector_store import (
    build_faiss_index,
    save_chunks,
    save_faiss_index,
)

router = APIRouter(tags=["upload"])


def validate_pdf_file(uploaded_file: UploadFile) -> None:
    if uploaded_file.content_type != "application/pdf":
        raise InvalidFileTypeError("Only PDF files are supported.")


@router.post(
    "/upload",
    response_model=UploadAndIndexResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_pdf(file: UploadFile = File(...)) -> UploadAndIndexResponse:
    try:
        validate_pdf_file(file)

        file_bytes = await file.read()

        extracted_text, page_count = extract_text_from_pdf_bytes(file_bytes)
        chunks = chunk_text(extracted_text)

        if not chunks:
            raise DocumentProcessingError(
                "No valid chunks could be generated from the extracted document text."
            )

        embeddings = generate_embeddings(chunks)

        if not embeddings:
            raise EmbeddingGenerationError(
                "No embeddings could be generated from the document chunks."
            )

        if len(chunks) != len(embeddings):
            raise DocumentProcessingError(
                "Mismatch between generated chunks and embeddings."
            )

        index = build_faiss_index(embeddings)

        save_faiss_index(index)
        save_chunks(chunks)

        return UploadAndIndexResponse(
            filename=file.filename or "unknown.pdf",
            content_type=file.content_type or "application/pdf",
            page_count=page_count,
            extracted_text_length=len(extracted_text),
            chunk_count=len(chunks),
            embedding_count=len(embeddings),
            status="indexed",
        )

    except InvalidFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except PDFExtractionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DocumentProcessingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except EmbeddingGenerationError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except VectorStoreError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc