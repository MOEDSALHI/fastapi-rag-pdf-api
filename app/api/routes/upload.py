from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.exceptions import InvalidFileTypeError, PDFExtractionError
from app.schemas.upload import UploadResponse
from app.services.pdf_extractor import extract_text_from_pdf_bytes

router = APIRouter(tags=["upload"])


def validate_pdf_file(uploaded_file: UploadFile) -> None:
    if uploaded_file.content_type != "application/pdf":
        raise InvalidFileTypeError("Only PDF files are supported.")


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    try:
        validate_pdf_file(file)

        file_bytes = await file.read()
        extracted_text, page_count = extract_text_from_pdf_bytes(file_bytes)

        return UploadResponse(
            filename=file.filename or "unknown.pdf",
            content_type=file.content_type or "application/pdf",
            extracted_text=extracted_text,
            extracted_text_length=len(extracted_text),
            page_count=page_count,
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