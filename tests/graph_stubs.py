"""Scripted stand-ins for running the knowledge_qa graph without a model.

Used by the graph tests in ``tests/agent`` and Tool 02's check tests in
``tests/tools``. The ``stub`` fixture replaces every model call; tool searches
are scripted with ``set_retrieval`` (Tool 01) and ``set_nice_retrieval`` (Tool 02).
"""

from __future__ import annotations

import pytest

from src.agent.knowledge_qa import nodes as qa_nodes
from src.agent.knowledge_qa.graph import build_knowledge_qa_graph
from src.agent.state import RetrievedChunk
from src.shared.errors import AgentUnavailableError
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_02_nice_guidance import nodes as tool_02_nodes
from src.tools.tool_02_nice_guidance import service as tool_02


def make_chunk(title: str = "Migraine", score: float = 0.8) -> RetrievedChunk:
    """Build one retrieved chunk."""
    return RetrievedChunk(
        chunk_id=f"chunk-{title}",
        document_id=f"document-{title}",
        title=title,
        text=f"{title} is managed with supportive care.",
        score=score,
    )


class StubModel:
    """A scripted stand-in for the model adapter."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.prompts: dict[str, list[str]] = {}
        self.keep_passages = True
        self.grounded = True
        self.answer = "Supportive care is indicated [1]."
        self.verify_raises = False
        self.unsupported: list[str] = []
        self.nice_verdict: dict = {
            "claims": [
                {
                    "claim": "Supportive care is indicated.",
                    "verdict": "agrees",
                    "nice_says": "Offer supportive care.",
                    "passages": [1],
                }
            ]
        }
        self.nice_raises = False

    def _record(self, label: str, messages) -> None:
        self.calls.append(label)
        self.prompts.setdefault(label, []).append(messages[0]["content"] if messages else "")

    async def structured(self, *, schema, label, messages=None, **_):
        self._record(label, messages)
        if label in {"rewrite_query", "broaden_query"}:
            return schema(search_query=f"{label}-result")
        if label == "grade_relevance":
            return schema(
                verdicts=[
                    {"passage": number, "keep": self.keep_passages, "reason": "stub"}
                    for number in range(1, 6)
                ]
            )
        if label == "verify_groundedness":
            if self.verify_raises:
                raise AgentUnavailableError("model down")
            return schema(grounded=self.grounded, unsupported_claims=self.unsupported)
        if label == "validate_against_nice":
            if self.nice_raises:
                raise AgentUnavailableError("model down")
            return schema.model_validate(self.nice_verdict)
        raise AssertionError(f"unexpected structured call: {label}")

    async def text(self, *, label, messages=None, **_):
        self._record(label, messages)
        return self.answer


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> StubModel:
    """Install the scripted model in place of the model adapter."""
    model = StubModel()
    monkeypatch.setattr(qa_nodes, "complete_structured", model.structured)
    monkeypatch.setattr(qa_nodes, "complete_text", model.text)
    monkeypatch.setattr(tool_02_nodes, "complete_structured", model.structured)
    return model


@pytest.fixture
def no_nice_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool 02 finds nothing unless a test scripts it, so no embedding model loads."""
    monkeypatch.setattr(tool_02, "retrieve", lambda _query, top_k=None: [])


def make_nice_chunk(text: str = "Offer metformin first line.", number: int = 1) -> RetrievedChunk:
    """Build one retrieved NICE chunk."""
    return RetrievedChunk(
        chunk_id=f"nice-chunk-{number}",
        document_id="nice-document",
        title="Type 2 diabetes in adults: management",
        text=text,
        score=0.7,
    )


def set_retrieval(monkeypatch: pytest.MonkeyPatch, *results: list[RetrievedChunk]):
    """Script successive retrieval results, repeating the last one."""
    sequence = list(results)

    def retrieve(_query, top_k=None):
        return sequence.pop(0) if len(sequence) > 1 else sequence[0]

    monkeypatch.setattr(tool_01, "retrieve", retrieve)


async def run(question: str = "How is migraine managed?") -> dict:
    """Invoke the compiled subgraph once."""
    graph = build_knowledge_qa_graph().compile()
    return await graph.ainvoke(
        {"session_id": "test-session", "question": question, "history": []}
    )


def set_nice_retrieval(monkeypatch: pytest.MonkeyPatch, *chunks: RetrievedChunk) -> list[str]:
    """Script tool 02's result and return the list of queries it receives."""
    queries: list[str] = []

    def retrieve(query, top_k=None):
        queries.append(query)
        return list(chunks)

    monkeypatch.setattr(tool_02, "retrieve", retrieve)
    return queries

