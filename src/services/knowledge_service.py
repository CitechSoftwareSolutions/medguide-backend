"""Business operations for medical knowledge entries."""

from uuid import UUID

from src.dto import CreateKnowledgeEntryRequest
from src.enums import KnowledgeType
from src.exceptions import (
    DuplicateKnowledgeEntryError,
    KnowledgeEntryNotFoundError,
)
from src.models import KnowledgeEntry
from src.repositories import find_by_id, find_by_title, list_all, save
from src.utils import current_utc_time, new_identifier


def create_knowledge_entry(payload: CreateKnowledgeEntryRequest) -> KnowledgeEntry:
    """Create an entry after applying domain rules."""
    if find_by_title(payload.title):
        raise DuplicateKnowledgeEntryError(
            f"A knowledge entry named '{payload.title}' already exists."
        )

    entry = KnowledgeEntry(
        id=new_identifier(),
        title=payload.title,
        summary=payload.summary,
        knowledge_type=payload.knowledge_type,
        tags=payload.tags,
        created_at=current_utc_time(),
    )
    return save(entry)


def list_knowledge_entries(
    knowledge_type: KnowledgeType | None = None,
) -> list[KnowledgeEntry]:
    """List entries, optionally limited to one type."""
    entries = list_all()
    if knowledge_type is None:
        return entries
    return [entry for entry in entries if entry.knowledge_type == knowledge_type]


def get_knowledge_entry(entry_id: UUID) -> KnowledgeEntry:
    """Get an entry or raise a meaningful not-found error."""
    entry = find_by_id(entry_id)
    if entry is None:
        raise KnowledgeEntryNotFoundError("No knowledge entry exists for this id.")
    return entry
