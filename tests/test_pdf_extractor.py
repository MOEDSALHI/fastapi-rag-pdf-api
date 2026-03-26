import pytest

from app.core.exceptions import PDFExtractionError
from app.services.pdf_extractor import extract_text_from_pdf_bytes


def test_extract_text_from_invalid_pdf_bytes_raises_error() -> None:
    invalid_bytes = b"this is not a real pdf"

    with pytest.raises(PDFExtractionError):
        extract_text_from_pdf_bytes(invalid_bytes)