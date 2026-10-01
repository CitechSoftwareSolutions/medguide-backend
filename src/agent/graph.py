"""The session graph: one question in, one answer out, with chat memory.

Flow::

    knowledge_qa -> record_turn -> END

Every question goes straight to the knowledge_qa graph; there is no routing
step. A question the knowledge base does not cover ends in knowledge_qa's
abstention.

This wrapper exists for memory. It is compiled with the checkpointer, so only
its small state (question, answer, history) is saved per session, while
knowledge_qa's working state (queries, chunks, retry counters) starts fresh on
every question.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.knowledge_qa.graph import get_knowledge_qa_graph
from src.agent.memory import get_checkpointer
from src.agent.state import SessionState, Turn, nice_not_run
from src.shared.config import get_settings


async def run_knowledge_qa(state: SessionState) -> dict:
    """Invoke the knowledge_qa graph and return only its result.

    Both state schemas carry a ``history`` key, and this graph's key is reduced
    by ``append_turns``; handing knowledge_qa's copy of the history straight back
    would append that list to itself and double the conversation on every turn.
    Naming the keys that may cross makes that impossible.
    """
    result = await get_knowledge_qa_graph().ainvoke(
        {
            "session_id": state["session_id"],
            "question": state["question"],
            "history": state.get("history", []),
        },
        # The retry counters in knowledge_qa already bound every path; this is
        # the backstop in case a future edge is added without one.
        config={"recursion_limit": get_settings().agent_recursion_limit},
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "confidence": result.get("confidence") or "abstained",
        "nice_validation": result.get("nice_validation") or nice_not_run(),
        "timings": result.get("timings") or [],
    }


async def record_turn(state: SessionState) -> dict:
    """Commit this exchange to session history.

    The state reducer handles appending and trimming, so this node only needs to
    hand over the two new turns.
    """
    return {
        "history": [
            Turn(role="user", content=state["question"]),
            Turn(role="assistant", content=state.get("answer", "")),
        ]
    }


def build_session_graph() -> StateGraph:
    graph = StateGraph(SessionState)
    graph.add_node("knowledge_qa", run_knowledge_qa)
    graph.add_node("record_turn", record_turn)
    graph.add_edge(START, "knowledge_qa")
    graph.add_edge("knowledge_qa", "record_turn")
    graph.add_edge("record_turn", END)
    return graph


@lru_cache
def get_session_graph():
    """Return the compiled session graph, compiled once per process."""
    return build_session_graph().compile(checkpointer=get_checkpointer())
