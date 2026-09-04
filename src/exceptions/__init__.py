"""Application errors and their HTTP handlers."""

from .errors import (
    ApplicationError,
    DuplicateKnowledgeEntryError,
    KnowledgeEntryNotFoundError,
)
from .handlers import register_exception_handlers

__all__ = [
    "ApplicationError",
    "DuplicateKnowledgeEntryError",
    "KnowledgeEntryNotFoundError",
    "register_exception_handlers",
]
