"""Repository tests using a fake SQLAlchemy session factory."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.enums import KnowledgeType
from src.exceptions import DuplicateKnowledgeEntryError
from src.models import KnowledgeEntry
from src.repositories import knowledge_repository


class FakeSession:
    """Small session fake that records repository interactions."""

    def __init__(self, *, commit_error: Exception | None = None) -> None:
        self.commit_error = commit_error
        self.added: list[KnowledgeEntry] = []
        self.committed = False
        self.rolled_back = False
        self.refreshed: list[KnowledgeEntry] = []

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def add(self, entry: KnowledgeEntry) -> None:
        self.added.append(entry)

    def commit(self) -> None:
        if self.commit_error is not None:
            raise self.commit_error
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def refresh(self, entry: KnowledgeEntry) -> None:
        self.refreshed.append(entry)


def make_entry() -> KnowledgeEntry:
    """Create an ORM object without persistence."""
    return KnowledgeEntry(
        id=uuid4(),
        title="Migraine",
        summary="A recurrent headache disorder.",
        knowledge_type=KnowledgeType.CONDITION,
        tags=[],
        created_at=datetime.now(UTC),
    )


def test_save_commits_and_refreshes_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """The repository owns transaction completion for a write."""
    session = FakeSession()
    monkeypatch.setattr(knowledge_repository, "SessionLocal", lambda: session)
    entry = make_entry()

    assert knowledge_repository.save(entry) is entry
    assert session.added == [entry]
    assert session.committed is True
    assert session.refreshed == [entry]


def test_save_maps_database_uniqueness_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A database integrity failure becomes the API's duplicate domain error."""
    from sqlalchemy.exc import IntegrityError

    session = FakeSession(commit_error=IntegrityError("statement", {}, Exception()))
    monkeypatch.setattr(knowledge_repository, "SessionLocal", lambda: session)

    with pytest.raises(DuplicateKnowledgeEntryError):
        knowledge_repository.save(make_entry())

    assert session.rolled_back is True
