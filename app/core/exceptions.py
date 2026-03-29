class AppException(Exception):
    """Base exception for the application."""


class InvalidFileTypeError(AppException):
    """Raised when the uploaded file type is not supported."""


class PDFExtractionError(AppException):
    """Raised when text extraction from a PDF fails."""


class EmbeddingGenerationError(AppException):
    """Raised when embedding generation fails."""


class LLMResponseGenerationError(AppException):
    """Raised when answer generation with the chat model fails."""


class VectorStoreError(AppException):
    """Raised when vector store operations fail."""


class DocumentProcessingError(AppException):
    """Raised when a document cannot be processed into valid chunks or embeddings."""
