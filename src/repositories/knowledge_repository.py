"""In-memory persistence for the sample API.

Replace this module with a database adapter when persistent storage is added.
"""

from uuid import UUID

from src.models import KnowledgeEntry

_entries: list[KnowledgeEntry] = []


def save(entry: KnowledgeEntry) -> KnowledgeEntry:
    """Store an entry and return it."""
    _entries.append(entry)
    return entry


def list_all() -> list[KnowledgeEntry]:
    """Return every stored entry without exposing the internal list."""
    return list(_entries)


def find_by_id(entry_id: UUID) -> KnowledgeEntry | None:
    """Find a single entry by identifier."""
    return next((entry for entry in _entries if entry.id == entry_id), None)


def find_by_title(title: str) -> KnowledgeEntry | None:
    """Find an entry by title, ignoring letter case."""
    normalized_title = title.casefold()
    return next(
        (entry for entry in _entries if entry.title.casefold() == normalized_title),
        None,
    )


def has_entries() -> bool:
    """Report whether the sample store already has data."""
    return bool(_entries)
