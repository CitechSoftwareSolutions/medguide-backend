"""The knowledge_qa graph: where the agent orchestrates Tool 01 and Tool 02.

Flow::

                    +-> tool_01.retrieve_knowledge_base -+
    rewrite_query --|                                    |-> grade_relevance
     broaden_query -+-> tool_02.retrieve_nice -----------+         |
          ^                                                        |
          +------------------ nothing relevant --------------------+
                                                                   |
          synthesize <-> verify_groundedness -> tool_02.validate_against_nice -> END
                                  |
    abstain <---------------------+----- retries used up

Both tools search with the same query in one step, and ``grade_relevance``
waits for both. Only Tool 01's passages are graded and written from; Tool 02's
passages wait in state for ``validate_against_nice``, which checks the grounded
answer and reports without changing it.

``abstain`` is reachable from both decision points, and both retry paths are
capped by counters in ``nodes.py``, so every route provably terminates.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.knowledge_qa import nodes
from src.agent.state import QaState
from src.tools.tool_01_knowledge_base import nodes as tool_01
from src.tools.tool_02_nice_guidance import nodes as tool_02


def build_knowledge_qa_graph() -> StateGraph:
    """Wire the answer-path nodes and both tools into an uncompiled graph."""
    graph = StateGraph(QaState)

    graph.add_node("rewrite_query", nodes.rewrite_query)
    graph.add_node("retrieve_knowledge_base", tool_01.retrieve_knowledge_base)
    graph.add_node("retrieve_nice", tool_02.retrieve_nice)
    graph.add_node("grade_relevance", nodes.grade_relevance)
    graph.add_node("broaden_query", nodes.broaden_query)
    graph.add_node("synthesize", nodes.synthesize)
    graph.add_node("verify_groundedness", nodes.verify_groundedness)
    graph.add_node("validate_against_nice", tool_02.validate_against_nice)
    graph.add_node("abstain", nodes.abstain)

    graph.add_edge(START, "rewrite_query")
    # Fan out: both tools search with the same query in the same step.
    for query_node in ("rewrite_query", "broaden_query"):
        graph.add_edge(query_node, "retrieve_knowledge_base")
        graph.add_edge(query_node, "retrieve_nice")
    # Join: grading starts only once both searches have finished.
    graph.add_edge(["retrieve_knowledge_base", "retrieve_nice"], "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        nodes.after_grade,
        {
            "synthesize": "synthesize",
            "broaden_query": "broaden_query",
            "abstain": "abstain",
        },
    )
    graph.add_edge("synthesize", "verify_groundedness")
    graph.add_conditional_edges(
        "verify_groundedness",
        nodes.after_verify,
        {
            "finish": "validate_against_nice",
            "synthesize": "synthesize",
            "abstain": "abstain",
        },
    )
    graph.add_edge("validate_against_nice", END)
    graph.add_edge("abstain", END)

    return graph


@lru_cache
def get_knowledge_qa_graph():
    """Return the compiled graph, compiled once per process.

    No checkpointer is passed here: the parent graph's checkpointer propagates
    into subgraphs automatically, so passing one would create a second,
    competing store for the same session.
    """
    return build_knowledge_qa_graph().compile()
