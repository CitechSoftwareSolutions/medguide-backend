"""HTTP endpoints for medical knowledge entries."""

from uuid import UUID

from fastapi import APIRouter, status

from src.controllers import create_entry, get_entry, import_guideline, list_entries
from src.dto import (
    CreateKnowledgeEntryRequest,
    ImportGuidelineRequest,
    ImportGuidelineResponse,
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


@router.post(
    "/import",
    response_model=ImportGuidelineResponse,
    status_code=status.HTTP_201_CREATED,
)
def import_guideline_document(
    payload: ImportGuidelineRequest,
) -> ImportGuidelineResponse:
    """Bulk-import a structured guideline document as knowledge entries.

    Splits a document into one entry per condition plus one overview entry for
    its shared workflow, red flags and best practices, then indexes each new
    one so it is immediately searchable. Re-posting the same document is
    safe: entries already present by title are reused rather than duplicated,
    in the knowledge table and in the search index alike.
    """
    return import_guideline(payload)


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
