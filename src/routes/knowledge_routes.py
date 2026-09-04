"""HTTP endpoints for medical knowledge entries."""

from uuid import UUID

from fastapi import APIRouter, status

from src.controllers import create_entry, get_entry, list_entries
from src.dto import (
    CreateKnowledgeEntryRequest,
    KnowledgeEntryListResponse,
    KnowledgeEntryResponse,
)
from src.enums import KnowledgeType

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("", response_model=KnowledgeEntryResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_entry(
    payload: CreateKnowledgeEntryRequest,
) -> KnowledgeEntryResponse:
    """Create a medical knowledge entry."""
    return create_entry(payload)


@router.get("", response_model=KnowledgeEntryListResponse)
def list_knowledge_entries(
    knowledge_type: KnowledgeType | None = None,
) -> KnowledgeEntryListResponse:
    """List entries, optionally filtered by their category."""
    return list_entries(knowledge_type)


@router.get("/{entry_id}", response_model=KnowledgeEntryResponse)
def get_knowledge_entry(
    entry_id: UUID,
) -> KnowledgeEntryResponse:
    """Get one entry by UUID."""
    return get_entry(entry_id)
