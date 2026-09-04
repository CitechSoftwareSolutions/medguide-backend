"""Time helpers."""

from datetime import UTC, datetime


def current_utc_time() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(UTC)
