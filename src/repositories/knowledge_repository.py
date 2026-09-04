"""SQLAlchemy persistence operations for medical knowledge entries."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.config import SessionLocal
from src.enums import KnowledgeType
from src.exceptions import DuplicateKnowledgeEntryError
from src.models import KnowledgeEntry


def save(entry: KnowledgeEntry) -> KnowledgeEntry:
    """Commit a new entry and return its refreshed ORM record."""
    with SessionLocal() as session:
        session.add(entry)
        try:
            session.commit()
        except IntegrityError as exc:
            session.rollback()
            raise DuplicateKnowledgeEntryError(
                f"A knowledge entry named '{entry.title}' already exists."
            ) from exc
        session.refresh(entry)
        return entry


def list_all(knowledge_type: KnowledgeType | None = None) -> list[KnowledgeEntry]:
    """Return every entry, optionally filtered by category."""
    with SessionLocal() as session:
        statement = select(KnowledgeEntry).order_by(KnowledgeEntry.title)
        if knowledge_type is not None:
            statement = statement.where(KnowledgeEntry.knowledge_type == knowledge_type)
        return list(session.scalars(statement))


def find_by_id(entry_id: UUID) -> KnowledgeEntry | None:
    """Find a single entry by identifier."""
    with SessionLocal() as session:
        return session.get(KnowledgeEntry, entry_id)


def find_by_title(title: str) -> KnowledgeEntry | None:
    """Find an entry by title, ignoring letter case."""
    with SessionLocal() as session:
        statement = select(KnowledgeEntry).where(
            KnowledgeEntry.title.ilike(title)
        )
        return session.scalar(statement)


def has_entries() -> bool:
    """Report whether the sample store already has data."""
    with SessionLocal() as session:
        return session.scalar(select(KnowledgeEntry.id).limit(1)) is not None
