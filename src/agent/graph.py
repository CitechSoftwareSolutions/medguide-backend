"""The supervisor graph that routes a request to one capability.

Flow::

    route -> [knowledge_qa subgraph] -> record_turn -> END
          \\-> decline_out_of_scope --/

With one capability the supervisor is thin by design. Its value is the seam:
tomorrow's capabilities are sibling subgraph nodes with one conditional edge
each, and the question-answering path does not change to accommodate them.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.memory import get_checkpointer
from src.agent.nodes import supervisor_nodes
from src.agent.state import SupervisorState
from src.agent.tools import registry
from src.agent.tools.knowledge_qa_tool import get_knowledge_qa_graph
from src.config import get_settings


async def run_knowledge_qa(state: SupervisorState) -> dict:
    """Invoke the question-answering subgraph and return only its result.

    The subgraph is called through this wrapper rather than added directly as a
    node so the boundary is explicit. Both state schemas carry a ``history``
    key, and the parent's key is reduced by ``append_turns``; handing the
    subgraph's copy of the history straight back would append that list to
    itself and double the conversation on every turn. Naming the three keys
    that may cross makes that impossible, and keeps the subgraph's working
    state (queries, chunks, retry counters) out of the session checkpoint.
    """
    result = await get_knowledge_qa_graph().ainvoke(
        {
            "session_id": state["session_id"],
            "question": state["question"],
            "history": state.get("history", []),
        },
        # The retry counters in qa_nodes already bound every path; this is the
        # backstop in case a future edge is added without one.
        config={"recursion_limit": get_settings().agent_recursion_limit},
    )
    return {
        "answer": result.get("answer", ""),
        "citations": result.get("citations") or [],
        "confidence": result.get("confidence") or "abstained",
    }


def build_supervisor_graph() -> StateGraph:
    """Wire the supervisor nodes and capability subgraphs into a graph."""
    graph = StateGraph(SupervisorState)

    graph.add_node("route", supervisor_nodes.route)
    graph.add_node(registry.KNOWLEDGE_QA, run_knowledge_qa)
    graph.add_node("decline_out_of_scope", supervisor_nodes.decline_out_of_scope)
    graph.add_node("record_turn", supervisor_nodes.record_turn)

    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        supervisor_nodes.after_route,
        {
            registry.KNOWLEDGE_QA: registry.KNOWLEDGE_QA,
            "decline_out_of_scope": "decline_out_of_scope",
        },
    )
    # Every capability converges on record_turn so history is written exactly
    # once, whatever path produced the answer.
    graph.add_edge(registry.KNOWLEDGE_QA, "record_turn")
    graph.add_edge("decline_out_of_scope", "record_turn")
    graph.add_edge("record_turn", END)

    return graph


@lru_cache
def get_supervisor_graph():
    """Return the compiled supervisor graph, compiled once per process."""
    return build_supervisor_graph().compile(checkpointer=get_checkpointer())
