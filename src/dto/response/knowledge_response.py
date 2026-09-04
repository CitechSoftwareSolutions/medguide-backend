"""Response DTOs for knowledge-entry endpoints."""

from pydantic import BaseModel

from src.models import KnowledgeEntry


class KnowledgeEntryResponse(BaseModel):
    """The standard response body for a single entry."""

    data: KnowledgeEntry


class KnowledgeEntryListResponse(BaseModel):
    """The standard response body for a collection of entries."""

    data: list[KnowledgeEntry]
    count: int
