"""The domain representation of a knowledge entry."""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from src.enums import KnowledgeType


class KnowledgeEntry(BaseModel):
    """A medical knowledge item stored by the application."""

    model_config = ConfigDict(frozen=True)

    id: UUID
    title: str
    summary: str
    knowledge_type: KnowledgeType
    tags: list[str] = Field(default_factory=list)
    created_at: datetime
