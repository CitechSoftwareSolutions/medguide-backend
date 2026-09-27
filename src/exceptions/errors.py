"""Expected errors raised by the application layer."""


class ApplicationError(Exception):
    """Base error that is safe to expose to an API client."""

    status_code = 400
    error_code = "application_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class InvalidDocumentError(ApplicationError):
    """Raised when an uploaded file is not a usable PDF."""

    status_code = 422
    error_code = "invalid_document"


class DuplicateDocumentError(ApplicationError):
    """Raised when an uploaded PDF is already in the knowledge base."""

    status_code = 409
    error_code = "duplicate_document"


class DocumentTooLargeError(ApplicationError):
    """Raised when an upload exceeds the configured size limit."""

    status_code = 413
    error_code = "document_too_large"


class AgentUnavailableError(ApplicationError):
    """Raised when the language model cannot be reached or refuses to serve."""

    status_code = 503
    error_code = "agent_unavailable"


class KnowledgeBaseEmptyError(ApplicationError):
    """Raised when a question arrives before any knowledge has been indexed."""

    status_code = 409
    error_code = "knowledge_base_empty"
