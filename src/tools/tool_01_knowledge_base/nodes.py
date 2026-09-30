"""The graph node Tool 01 adds to the agent's knowledge_qa graph."""

from __future__ import annotations

import logging

import anyio

from src.agent.state import QaState
from src.tools.tool_01_knowledge_base import service

logger = logging.getLogger(__name__)


async def retrieve_knowledge_base(state: QaState) -> dict:
    """Search the knowledge base. Deterministic; no model involved.

    The embedding call is CPU-bound and blocking, so it runs on a worker thread
    to keep the event loop free.
    """
    query = state.get("search_query") or state["question"]
    chunks = await anyio.to_thread.run_sync(service.retrieve, query)
    logger.debug("Retrieved %d knowledge base chunks for %r", len(chunks), query)
    return {
        "chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }
