"""Identifier helpers."""

from uuid import UUID, uuid4


def new_identifier() -> UUID:
    """Create an identifier for a newly stored entry."""
    return uuid4()
