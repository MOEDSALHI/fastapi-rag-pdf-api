from pydantic import BaseModel


class UploadResponse(BaseModel):
    filename: str
    content_type: str
    extracted_text: str
    extracted_text_length: int
    page_count: int