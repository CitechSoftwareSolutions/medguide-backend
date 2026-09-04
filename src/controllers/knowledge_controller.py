"""Functions that shape service results into API response DTOs."""

from uuid import UUID

from src.dto import (
    CreateKnowledgeEntryRequest,
    KnowledgeEntryData,
    KnowledgeEntryListResponse,
    KnowledgeEntryResponse,
)
from src.enums import KnowledgeType
from src.services import knowledge_service


def create_entry(payload: CreateKnowledgeEntryRequest) -> KnowledgeEntryResponse:
    """Create one knowledge entry."""
    entry = knowledge_service.create_knowledge_entry(payload)
    return KnowledgeEntryResponse(data=KnowledgeEntryData.model_validate(entry))


def list_entries(
    knowledge_type: KnowledgeType | None = None,
) -> KnowledgeEntryListResponse:
    """List knowledge entries."""
    entries = knowledge_service.list_knowledge_entries(knowledge_type)
    response_entries = [KnowledgeEntryData.model_validate(entry) for entry in entries]
    return KnowledgeEntryListResponse(data=response_entries, count=len(response_entries))


def get_entry(entry_id: UUID) -> KnowledgeEntryResponse:
    """Get one knowledge entry."""
    entry = knowledge_service.get_knowledge_entry(entry_id)
    return KnowledgeEntryResponse(data=KnowledgeEntryData.model_validate(entry))
