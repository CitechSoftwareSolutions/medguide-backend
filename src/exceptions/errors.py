"""Expected errors raised by the application layer."""


class ApplicationError(Exception):
    """Base error that is safe to expose to an API client."""

    status_code = 400
    error_code = "application_error"

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


class KnowledgeEntryNotFoundError(ApplicationError):
    """Raised when an entry identifier does not match an entry."""

    status_code = 404
    error_code = "knowledge_entry_not_found"


class DuplicateKnowledgeEntryError(ApplicationError):
    """Raised when a title already exists in the knowledge base."""

    status_code = 409
    error_code = "duplicate_knowledge_entry"
