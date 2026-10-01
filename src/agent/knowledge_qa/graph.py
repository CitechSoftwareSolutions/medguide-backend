"""The knowledge_qa graph: where the agent orchestrates Tool 01 and Tool 02.

Flow::

                    +-> tool_01.retrieve_knowledge_base -+
    rewrite_query --|                                    |-> join_searches
     broaden_query -+-> tool_02.retrieve_nice -----------+         |
          ^                                                        |
          +---------- Tool 01 found nothing above the floor -------+
                                                                   |
          synthesize <-> verify_groundedness -> tool_02.validate_against_nice -> END
                                  |
    abstain <---------------------+----- retries used up

Both tools search with the same query in one step, and ``join_searches`` waits
for both. Only Tool 01's passages are written from; Tool 02's passages wait in
state for ``validate_against_nice``, which checks the grounded answer and
reports without changing it.

A first question makes three model calls: synthesize, verify_groundedness and
validate_against_nice. A follow-up adds ``rewrite_query``, and a search that
finds nothing adds ``broaden_query``.

``abstain`` is reachable from both decision points, and both retry paths are
capped by counters in ``nodes.py``, so every route provably terminates.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.knowledge_qa import nodes
from src.agent.state import QaState
from src.agent.timing import timed
from src.tools.tool_01_knowledge_base import nodes as tool_01
from src.tools.tool_02_nice_guidance import nodes as tool_02


def build_knowledge_qa_graph() -> StateGraph:
    """Wire the answer-path nodes and both tools into an uncompiled graph."""
    graph = StateGraph(QaState)

    # Every node is timed; the label is what appears in logs and the response.
    steps = {
        "rewrite_query": ("rewrite_query", nodes.rewrite_query),
        "retrieve_knowledge_base": (
            "tool_01.retrieve_knowledge_base",
            tool_01.retrieve_knowledge_base,
        ),
        "retrieve_nice": ("tool_02.retrieve_nice", tool_02.retrieve_nice),
        "join_searches": ("join_searches", nodes.join_searches),
        "broaden_query": ("broaden_query", nodes.broaden_query),
        "synthesize": ("synthesize", nodes.synthesize),
        "verify_groundedness": ("verify_groundedness", nodes.verify_groundedness),
        "validate_against_nice": (
            "tool_02.validate_against_nice",
            tool_02.validate_against_nice,
        ),
        "abstain": ("abstain", nodes.abstain),
    }
    for name, (label, node) in steps.items():
        graph.add_node(name, timed(label, node))

    graph.add_edge(START, "rewrite_query")
    # Fan out: both tools search with the same query in the same step.
    for query_node in ("rewrite_query", "broaden_query"):
        graph.add_edge(query_node, "retrieve_knowledge_base")
        graph.add_edge(query_node, "retrieve_nice")
    # Join: the answer path continues only once both searches have finished.
    graph.add_edge(["retrieve_knowledge_base", "retrieve_nice"], "join_searches")

    graph.add_conditional_edges(
        "join_searches",
        nodes.after_search,
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
