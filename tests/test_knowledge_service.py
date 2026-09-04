"""Unit tests for function-based knowledge service rules."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.dto import CreateKnowledgeEntryRequest
from src.enums import KnowledgeType
from src.exceptions import DuplicateKnowledgeEntryError, KnowledgeEntryNotFoundError
from src.models import KnowledgeEntry
from src.services import knowledge_service


def make_payload() -> CreateKnowledgeEntryRequest:
    """Build valid input shared by service tests."""
    return CreateKnowledgeEntryRequest(
        title="Migraine",
        summary="A recurrent headache disorder.",
        knowledge_type=KnowledgeType.CONDITION,
        tags=["neurology"],
    )


def make_entry() -> KnowledgeEntry:
    """Build an ORM object without requiring a database connection."""
    return KnowledgeEntry(
        id=uuid4(),
        title="Migraine",
        summary="A recurrent headache disorder.",
        knowledge_type=KnowledgeType.CONDITION,
        tags=["neurology"],
        created_at=datetime.now(UTC),
    )


def test_create_entry_uses_repository_save(monkeypatch: pytest.MonkeyPatch) -> None:
    """A new entry is created when its title is available."""
    saved_entries: list[KnowledgeEntry] = []
    monkeypatch.setattr(knowledge_service, "find_by_title", lambda _: None)
    monkeypatch.setattr(
        knowledge_service,
        "save",
        lambda entry: saved_entries.append(entry) or entry,
    )

    entry = knowledge_service.create_knowledge_entry(make_payload())

    assert entry.title == "Migraine"
    assert saved_entries == [entry]


def test_create_entry_rejects_duplicate_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """The service returns a meaningful error before saving a duplicate."""
    monkeypatch.setattr(knowledge_service, "find_by_title", lambda _: make_entry())

    with pytest.raises(DuplicateKnowledgeEntryError):
        knowledge_service.create_knowledge_entry(make_payload())


def test_get_entry_raises_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing records are translated to a domain error."""
    monkeypatch.setattr(knowledge_service, "find_by_id", lambda _: None)

    with pytest.raises(KnowledgeEntryNotFoundError):
        knowledge_service.get_knowledge_entry(uuid4())


def test_list_entries_passes_category_to_repository(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Filtering remains a repository concern while the service forwards intent."""
    received_types: list[KnowledgeType | None] = []
    monkeypatch.setattr(
        knowledge_service,
        "list_all",
        lambda knowledge_type: received_types.append(knowledge_type) or [],
    )

    assert knowledge_service.list_knowledge_entries(KnowledgeType.MEDICATION) == []
    assert received_types == [KnowledgeType.MEDICATION]
