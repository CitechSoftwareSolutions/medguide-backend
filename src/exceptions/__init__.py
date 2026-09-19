"""Application errors and their HTTP handlers."""

from .errors import (
    AgentUnavailableError,
    ApplicationError,
    DuplicateKnowledgeEntryError,
    KnowledgeBaseEmptyError,
    KnowledgeEntryNotFoundError,
)
from .handlers import register_exception_handlers

__all__ = [
    "AgentUnavailableError",
    "ApplicationError",
    "DuplicateKnowledgeEntryError",
    "KnowledgeBaseEmptyError",
    "KnowledgeEntryNotFoundError",
    "register_exception_handlers",
]
