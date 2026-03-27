from pydantic import BaseModel


class UploadAndIndexResponse(BaseModel):
    filename: str
    content_type: str
    page_count: int
    extracted_text_length: int
    chunk_count: int
    embedding_count: int
    status: str