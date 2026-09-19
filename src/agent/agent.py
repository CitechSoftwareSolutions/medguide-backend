"""The public entrypoint to the agent.

Everything outside ``src/agent`` goes through :func:`answer_question`, so the
graph, the model adapter and the memory backend all stay replaceable without
touching the service, controller or route layers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from src.agent.graph import get_supervisor_graph
from src.agent.memory import resolve_session_id, session_config, touch
from src.agent.state import Citation, Confidence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentAnswer:
    """What the agent produced for one question."""

    session_id: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    confidence: Confidence = "abstained"
    route: str = ""


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
    )
