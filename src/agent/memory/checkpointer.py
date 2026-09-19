"""The conversation checkpointer.

Isolated in its own module so moving session memory into PostgreSQL later is a
one-line change here rather than an edit to the graph.

**Deployment constraint:** ``InMemorySaver`` is per-process. Under more than one
uvicorn worker, consecutive requests for the same session land on different
processes at random and memory appears to vanish intermittently. Run with
``--workers 1`` until this is backed by a database.
"""

from functools import lru_cache

from langgraph.checkpoint.memory import InMemorySaver


@lru_cache
def get_checkpointer() -> InMemorySaver:
    """Return the one process-wide checkpointer."""
    return InMemorySaver()
