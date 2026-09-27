"""Application errors and their HTTP handlers."""

from .errors import (
    AgentUnavailableError,
    ApplicationError,
    DocumentTooLargeError,
    DuplicateDocumentError,
    InvalidDocumentError,
    KnowledgeBaseEmptyError,
)
from .handlers import register_exception_handlers

__all__ = [
    "AgentUnavailableError",
    "ApplicationError",
    "DocumentTooLargeError",
    "DuplicateDocumentError",
    "InvalidDocumentError",
    "KnowledgeBaseEmptyError",
    "register_exception_handlers",
]
