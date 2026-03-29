from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.exceptions import (
    PDFExtractionError,
    VectorStoreError,
)

from app.main import app

client = TestClient(app)


def test_upload_pdf_indexes_document_and_returns_200() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with (
        patch(
            "app.api.routes.upload.extract_text_from_pdf_bytes",
            return_value=("This is extracted text.", 2),
        ),
        patch(
            "app.api.routes.upload.chunk_text",
            return_value=["chunk 1", "chunk 2"],
        ),
        patch(
            "app.api.routes.upload.generate_embeddings",
            return_value=[[0.1, 0.2], [0.3, 0.4]],
        ),
        patch(
            "app.api.routes.upload.build_faiss_index",
            return_value="fake-index",
        ),
        patch("app.api.routes.upload.save_faiss_index") as mock_save_index,
        patch("app.api.routes.upload.save_chunks") as mock_save_chunks,
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.pdf",
        "content_type": "application/pdf",
        "page_count": 2,
        "extracted_text_length": 23,
        "chunk_count": 2,
        "embedding_count": 2,
        "status": "indexed",
    }

    mock_save_index.assert_called_once_with("fake-index")
    mock_save_chunks.assert_called_once_with(["chunk 1", "chunk 2"])


def test_upload_pdf_returns_400_for_non_pdf_file() -> None:
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Only PDF files are supported."}


def test_upload_pdf_returns_422_when_pdf_has_no_extractable_text() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with patch(
        "app.api.routes.upload.extract_text_from_pdf_bytes",
        side_effect=PDFExtractionError(
            "No extractable text was found in the provided PDF."
        ),
    ):
        response = client.post(
            "/upload",
            files={"file": ("empty.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "No extractable text was found in the provided PDF."
    }


def test_upload_pdf_returns_422_when_no_chunks_are_generated() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with (
        patch(
            "app.api.routes.upload.extract_text_from_pdf_bytes",
            return_value=("This is extracted text.", 2),
        ),
        patch(
            "app.api.routes.upload.chunk_text",
            return_value=[],
        ),
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "No valid chunks could be generated from the extracted document text."
    }


def test_upload_pdf_returns_502_when_embedding_generation_returns_empty() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with (
        patch(
            "app.api.routes.upload.extract_text_from_pdf_bytes",
            return_value=("This is extracted text.", 2),
        ),
        patch(
            "app.api.routes.upload.chunk_text",
            return_value=["chunk 1", "chunk 2"],
        ),
        patch(
            "app.api.routes.upload.generate_embeddings",
            return_value=[],
        ),
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 502
    assert response.json() == {
        "detail": "No embeddings could be generated from the document chunks."
    }


def test_upload_pdf_returns_422_when_chunks_and_embeddings_count_mismatch() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with (
        patch(
            "app.api.routes.upload.extract_text_from_pdf_bytes",
            return_value=("This is extracted text.", 2),
        ),
        patch(
            "app.api.routes.upload.chunk_text",
            return_value=["chunk 1", "chunk 2"],
        ),
        patch(
            "app.api.routes.upload.generate_embeddings",
            return_value=[[0.1, 0.2]],
        ),
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "Mismatch between generated chunks and embeddings."
    }


def test_upload_pdf_returns_500_when_vector_store_fails() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with (
        patch(
            "app.api.routes.upload.extract_text_from_pdf_bytes",
            return_value=("This is extracted text.", 2),
        ),
        patch(
            "app.api.routes.upload.chunk_text",
            return_value=["chunk 1"],
        ),
        patch(
            "app.api.routes.upload.generate_embeddings",
            return_value=[[0.1, 0.2]],
        ),
        patch(
            "app.api.routes.upload.build_faiss_index",
            side_effect=VectorStoreError(
                "An error occurred while building the FAISS index."
            ),
        ),
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "An error occurred while building the FAISS index."
    }
