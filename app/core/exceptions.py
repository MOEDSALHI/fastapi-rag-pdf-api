class AppException(Exception):
    """Base exception for the application."""


class InvalidFileTypeError(AppException):
    """Raised when the uploaded file type is not supported."""


class PDFExtractionError(AppException):
    """Raised when text extraction from a PDF fails."""