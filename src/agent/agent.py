"""The public entrypoint to the agent.

The route calls :func:`answer_clinical_question`; everything else about the
agent (graphs, model adapter, memory) stays replaceable behind it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from src.agent.memory import resolve_session_id, session_config, touch
from src.agent.state import Citation, Confidence, NiceValidation, nice_not_run
from src.agent.supervisor.graph import get_supervisor_graph
from src.shared.errors import KnowledgeBaseEmptyError
from src.tools.tool_01_knowledge_base import service as knowledge_base

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentAnswer:
    """What the agent produced for one question."""

    session_id: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: Confidence = "abstained"
    route: str = ""
    nice_validation: NiceValidation = field(default_factory=nice_not_run)


async def answer_clinical_question(
    question: str,
    session_id: UUID | None = None,
) -> AgentAnswer:
    """Answer a question, refusing early when the knowledge base is empty.

    The empty-index case is checked here rather than inside the graph: it is a
    configuration problem, not a retrieval outcome, and it deserves a distinct
    status code instead of an abstention that looks like a content gap.
    """
    if knowledge_base.stats()["chunk_count"] == 0:
        raise KnowledgeBaseEmptyError(
            "No knowledge has been indexed yet, so no question can be answered."
        )
    return await answer_question(question, session_id)


async def answer_question(
    question: str,
    session_id: UUID | str | None = None,
) -> AgentAnswer:
    """Answer one clinical question within a chat session.

    Memory is keyed on ``session_id``: pass the value returned by a previous
    call to continue that conversation, or omit it to start a new one.
    """
    resolved = resolve_session_id(session_id)
    touch(resolved)

    graph = get_supervisor_graph()
    final_state = await graph.ainvoke(
        {
            "session_id": resolved,
            "question": question,
            # The reducer merges this with whatever the checkpointer restored,
            # so an empty list here does not erase earlier turns.
            "history": [],
        },
        config=session_config(resolved),
    )

    return AgentAnswer(
        session_id=resolved,
        answer=final_state.get("answer", ""),
        citations=list(final_state.get("citations") or []),
        confidence=final_state.get("confidence") or "abstained",
        route=final_state.get("route", ""),
        nice_validation=final_state.get("nice_validation") or nice_not_run(),
    )
