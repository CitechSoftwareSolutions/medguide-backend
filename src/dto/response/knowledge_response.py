"""Response DTOs for knowledge-entry endpoints."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.enums import KnowledgeType


class KnowledgeEntryData(BaseModel):
    """The public, serializable representation of one ORM record."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    summary: str
    knowledge_type: KnowledgeType
    tags: list[str] = Field(default_factory=list)
    created_at: datetime


class KnowledgeEntryResponse(BaseModel):
    """The standard response body for a single entry."""

    data: KnowledgeEntryData


class KnowledgeEntryListResponse(BaseModel):
    """The standard response body for a collection of entries."""

    data: list[KnowledgeEntryData]
    count: int
