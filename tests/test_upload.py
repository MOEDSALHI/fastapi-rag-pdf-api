# from unittest.mock import patch

# from fastapi.testclient import TestClient

# from app.main import app

# client = TestClient(app)


# def test_upload_pdf_returns_200_for_valid_pdf() -> None:
#     fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

#     with patch(
#         "app.api.routes.upload.extract_text_from_pdf_bytes",
#         return_value=("This is extracted text.", 2),
#     ):
#         response = client.post(
#             "/upload",
#             files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
#         )

#     assert response.status_code == 200
#     assert response.json() == {
#         "filename": "sample.pdf",
#         "content_type": "application/pdf",
#         "extracted_text": "This is extracted text.",
#         "extracted_text_length": 23,
#         "page_count": 2,
#     }


# def test_upload_pdf_returns_400_for_non_pdf_file() -> None:
#     response = client.post(
#         "/upload",
#         files={"file": ("sample.txt", b"hello world", "text/plain")},
#     )

#     assert response.status_code == 400
#     assert response.json() == {
#         "detail": "Only PDF files are supported."
#     }


# def test_upload_pdf_returns_422_when_extraction_fails() -> None:
#     fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

#     with patch(
#         "app.api.routes.upload.extract_text_from_pdf_bytes",
#         side_effect=Exception("unexpected low-level error"),
#     ):
#         response = client.post(
#             "/upload",
#             files={"file": ("broken.pdf", fake_pdf_bytes, "application/pdf")},
#         )

#     assert response.status_code == 500
    

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.core.exceptions import PDFExtractionError
from app.main import app

client = TestClient(app)


def test_upload_pdf_returns_200_for_valid_pdf() -> None:
    fake_pdf_bytes = b"%PDF-1.4 fake pdf content"

    with patch(
        "app.api.routes.upload.extract_text_from_pdf_bytes",
        return_value=("This is extracted text.", 2),
    ):
        response = client.post(
            "/upload",
            files={"file": ("sample.pdf", fake_pdf_bytes, "application/pdf")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "filename": "sample.pdf",
        "content_type": "application/pdf",
        "extracted_text": "This is extracted text.",
        "extracted_text_length": 23,
        "page_count": 2,
    }


def test_upload_pdf_returns_400_for_non_pdf_file() -> None:
    response = client.post(
        "/upload",
        files={"file": ("sample.txt", b"hello world", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Only PDF files are supported."
    }


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