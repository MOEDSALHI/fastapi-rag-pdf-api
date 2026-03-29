from io import BytesIO

from pypdf import PdfReader

from app.core.exceptions import PDFExtractionError


def extract_text_from_pdf_bytes(file_bytes: bytes) -> tuple[str, int]:
    """
    Extract text from a PDF represented as bytes.

    Returns:
        tuple[str, int]: extracted text and page count
    """
    try:
        pdf_stream = BytesIO(file_bytes)
        reader = PdfReader(pdf_stream)

        page_count = len(reader.pages)
        extracted_pages: list[str] = []

        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())

        extracted_text = "\n\n".join(extracted_pages).strip()

        if not extracted_text:
            raise PDFExtractionError(
                "No extractable text was found in the provided PDF."
            )

        return extracted_text, page_count

    except PDFExtractionError:
        raise
    except Exception as exc:
        raise PDFExtractionError(
            "An error occurred while extracting text from the PDF."
        ) from exc
