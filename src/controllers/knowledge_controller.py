"""Functions that shape service results into API response DTOs."""

from uuid import UUID

from src.dto import (
    CreateKnowledgeEntryRequest,
    KnowledgeEntryListResponse,
    KnowledgeEntryResponse,
)
from src.enums import KnowledgeType
from src.services import knowledge_service


def create_entry(payload: CreateKnowledgeEntryRequest) -> KnowledgeEntryResponse:
    """Create one knowledge entry."""
    return KnowledgeEntryResponse(data=knowledge_service.create_knowledge_entry(payload))


def list_entries(
    knowledge_type: KnowledgeType | None = None,
) -> KnowledgeEntryListResponse:
    """List knowledge entries."""
    entries = knowledge_service.list_knowledge_entries(knowledge_type)
    return KnowledgeEntryListResponse(data=entries, count=len(entries))


def get_entry(entry_id: UUID) -> KnowledgeEntryResponse:
    """Get one knowledge entry."""
    return KnowledgeEntryResponse(data=knowledge_service.get_knowledge_entry(entry_id))
