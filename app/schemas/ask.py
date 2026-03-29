from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question about the PDF")
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of similar chunks to retrieve",
    )


class AskResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: list[str]
