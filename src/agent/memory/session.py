"""Chat session identity and lifetime.

Sessions are in-process only for now. Memory persists for the whole chat session
but nothing is written to a database, so a restart clears it.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any
from uuid import UUID

from src.agent.memory.checkpointer import get_checkpointer
from src.config import get_settings
from src.utils import new_identifier

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_last_seen: dict[str, float] = {}


def resolve_session_id(session_id: UUID | str | None) -> str:
    """Return the caller's session identifier, minting one when absent."""
    return str(session_id) if session_id else str(new_identifier())


def session_config(session_id: str) -> dict[str, Any]:
    """Build the LangGraph config that binds an invocation to a session.

    ``thread_id`` is what makes the checkpointer reload this session's history,
    so it is the whole mechanism behind cross-turn memory.
    """
    return {
        "configurable": {"thread_id": session_id},
        "recursion_limit": get_settings().agent_recursion_limit,
    }


def touch(session_id: str) -> None:
    """Record activity and evict sessions that have gone idle.

    ``InMemorySaver`` never evicts on its own, so without this the checkpoint
    store grows for the lifetime of the process. Sweeping on write keeps it
    bounded without a background task.
    """
    settings = get_settings()
    now = time.monotonic()
    cutoff = now - settings.session_ttl_seconds

    with _lock:
        _last_seen[session_id] = now
        expired = [key for key, seen in _last_seen.items() if seen < cutoff]
        for key in expired:
            del _last_seen[key]

    for key in expired:
        _forget(key)

    if expired:
        logger.info("Evicted %d idle session(s)", len(expired))


def _forget(session_id: str) -> None:
    """Drop one session's checkpoints, tolerating older LangGraph versions."""
    checkpointer = get_checkpointer()
    delete_thread = getattr(checkpointer, "delete_thread", None)
    if delete_thread is None:
        # Older releases expose no deletion hook; the identifier is dropped from
        # the registry either way, so this leaks only the stored checkpoint.
        return
    try:
        delete_thread(session_id)
    except Exception:  # pragma: no cover - eviction must never fail a request
        logger.warning("Could not evict session %s", session_id, exc_info=True)


def active_session_count() -> int:
    """Report how many sessions are currently tracked."""
    with _lock:
        return len(_last_seen)
