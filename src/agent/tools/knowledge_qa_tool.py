"""The knowledge question-answering capability, as a compiled subgraph.

Flow::

    rewrite_query -> retrieve -> grade_relevance -> synthesize
                        ^              |               |
                  broaden_query <------+         verify_groundedness
                                       |               |
                                    abstain <----------+

``abstain`` is reachable from both decision points, and both retry paths are
capped by counters in the node module, so every route provably terminates.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import qa_nodes
from src.agent.state import QaState
from src.agent.tools.registry import KNOWLEDGE_QA_SPEC

TOOL_SPEC = KNOWLEDGE_QA_SPEC


def build_knowledge_qa_graph() -> StateGraph:
    """Wire the question-answering nodes into an uncompiled graph."""
    graph = StateGraph(QaState)

    graph.add_node("rewrite_query", qa_nodes.rewrite_query)
    graph.add_node("retrieve", qa_nodes.retrieve)
    graph.add_node("grade_relevance", qa_nodes.grade_relevance)
    graph.add_node("broaden_query", qa_nodes.broaden_query)
    graph.add_node("synthesize", qa_nodes.synthesize)
    graph.add_node("verify_groundedness", qa_nodes.verify_groundedness)
    graph.add_node("abstain", qa_nodes.abstain)

    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "grade_relevance")

    graph.add_conditional_edges(
        "grade_relevance",
        qa_nodes.after_grade,
        {
            "synthesize": "synthesize",
            "broaden_query": "broaden_query",
            "abstain": "abstain",
        },
    )
    graph.add_edge("broaden_query", "retrieve")
    graph.add_edge("synthesize", "verify_groundedness")
    graph.add_conditional_edges(
        "verify_groundedness",
        qa_nodes.after_verify,
        {
            "finish": END,
            "synthesize": "synthesize",
            "abstain": "abstain",
        },
    )
    graph.add_edge("abstain", END)

    return graph


@lru_cache
def get_knowledge_qa_graph():
    """Return the compiled subgraph, compiled once per process.

    No checkpointer is passed here: the parent graph's checkpointer propagates
    into subgraphs automatically, so passing one would create a second,
    competing store for the same session.
    """
    return build_knowledge_qa_graph().compile()
