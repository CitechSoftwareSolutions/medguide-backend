"""Tests for the knowledge_qa graph's answer path.

Every model call is stubbed. What is under test is the graph's control flow: when
it answers, when it retries, and when it refuses. Tool 02's check has its own
tests in ``tests/tools/test_tool_02_validation.py``.
"""

from __future__ import annotations

import pytest

from src.agent.knowledge_qa import nodes as qa_nodes
from src.shared.errors import AgentUnavailableError
from src.tools.tool_01_knowledge_base import service as tool_01
from tests.graph_stubs import StubModel, make_chunk, run, set_retrieval

pytestmark = pytest.mark.usefixtures("no_nice_guidance")


async def test_answers_and_cites_when_passages_support_it(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path returns the answer with resolved citations."""
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["answer"] == "Supportive care is indicated [1]."
    assert not state.get("abstained")
    assert state["confidence"] == "high"
    assert [citation["marker"] for citation in state["citations"]] == [1]
    assert state["citations"][0]["chunk_id"] == "chunk-Migraine"


async def test_broadens_once_then_abstains_when_nothing_is_relevant(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fruitless search is retried exactly once before refusing."""
    stub.keep_passages = False
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["abstained"] is True
    assert state["confidence"] == "abstained"
    assert state["citations"] == []
    assert state["retrieval_attempts"] == qa_nodes.MAX_RETRIEVAL_ATTEMPTS
    assert stub.calls.count("broaden_query") == 1
    assert "synthesize" not in stub.calls, "nothing should be written without support"


async def test_recovers_when_the_broadened_query_finds_something(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry path can succeed rather than only leading to abstention."""
    call_count = {"value": 0}

    def retrieve(_query, top_k=None):
        call_count["value"] += 1
        # Nothing on the first pass, a usable chunk once the query is broadened.
        return [] if call_count["value"] == 1 else [make_chunk()]

    monkeypatch.setattr(tool_01, "retrieve", retrieve)

    state = await run()

    assert not state.get("abstained")
    assert state["answer"] == "Supportive care is indicated [1]."
    assert stub.calls.count("broaden_query") == 1


async def test_retries_synthesis_once_then_abstains_when_ungrounded(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An answer that fails the groundedness audit is never returned."""
    stub.grounded = False
    stub.unsupported = ["Give 500 mg every four hours."]
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["abstained"] is True
    assert state["confidence"] == "abstained"
    assert stub.calls.count("synthesize") == qa_nodes.MAX_SYNTHESIS_ATTEMPTS
    assert "500 mg" not in state["answer"]


async def test_second_synthesis_is_told_what_was_unsupported(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The retry prompt names the rejected claims so it can correct them."""
    prompts: list[str] = []

    async def text(*, messages, label, **_):
        prompts.append(messages[0]["content"])
        return stub.answer

    monkeypatch.setattr(qa_nodes, "complete_text", text)
    stub.grounded = False
    stub.unsupported = ["Give 500 mg every four hours."]
    set_retrieval(monkeypatch, [make_chunk()])

    await run()

    assert len(prompts) == 2
    assert "500 mg" not in prompts[0]
    assert "Give 500 mg every four hours." in prompts[1]


async def test_abstains_when_the_groundedness_check_is_unavailable(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit fails closed: an unverified answer is not emitted."""
    stub.verify_raises = True
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["abstained"] is True
    assert state["answer"] != "Supportive care is indicated [1]."


async def test_grading_outage_keeps_retrieved_passages(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A grader outage costs precision, not availability."""

    async def structured(*, schema, label, **kwargs):
        if label == "grade_relevance":
            raise AgentUnavailableError("model down")
        return await stub.structured(schema=schema, label=label, **kwargs)

    monkeypatch.setattr(qa_nodes, "complete_structured", structured)
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert not state.get("abstained")
    assert state["citations"]


async def test_rewrite_outage_falls_back_to_the_raw_question(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rewriting is an accuracy aid, so its failure must not fail the request."""
    queries: list[str] = []

    async def structured(*, schema, label, **kwargs):
        if label == "rewrite_query":
            raise AgentUnavailableError("model down")
        return await stub.structured(schema=schema, label=label, **kwargs)

    def retrieve(query, top_k=None):
        queries.append(query)
        return [make_chunk()]

    monkeypatch.setattr(qa_nodes, "complete_structured", structured)
    monkeypatch.setattr(tool_01, "retrieve", retrieve)

    state = await run("How is migraine managed?")

    assert queries == ["How is migraine managed?"]
    assert not state.get("abstained")


@pytest.mark.parametrize(
    ("answer", "expected_markers"),
    [
        ("Both apply [1][2].", [1, 2]),
        ("Out of range [9].", []),
        ("Repeated [1] and again [1].", [1]),
        ("No citation at all.", []),
    ],
)
async def test_citations_resolve_only_markers_the_answer_used(
    stub: StubModel,
    monkeypatch: pytest.MonkeyPatch,
    answer: str,
    expected_markers: list[int],
) -> None:
    """Provenance reflects the answer text, and bad markers are dropped."""
    stub.answer = answer
    set_retrieval(monkeypatch, [make_chunk("Migraine"), make_chunk("Aspirin")])

    state = await run()

    assert [citation["marker"] for citation in state["citations"]] == expected_markers


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.9, "high"), (0.5, "medium"), (0.4, "low")],
)
async def test_confidence_tracks_match_strength(
    stub: StubModel,
    monkeypatch: pytest.MonkeyPatch,
    score: float,
    expected: str,
) -> None:
    """Confidence is derived from the strength of the supporting match."""
    set_retrieval(monkeypatch, [make_chunk(score=score)])

    state = await run()

    assert state["confidence"] == expected


async def test_verifier_objection_to_passage_text_is_overruled(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A claim copied from a passage stands even when the verifier flags it."""
    stub.grounded = False
    stub.unsupported = ["Migraine is managed with supportive care [1]."]
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state.get("abstained") is not True
    assert state["answer"] == stub.answer
    assert stub.calls.count("synthesize") == 1


async def test_verifier_objection_to_an_altered_dose_stands(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching words do not excuse a number the passages never state."""
    chunk = make_chunk()
    chunk["text"] = "Propranolol 40-320 mg/day in divided doses."
    stub.grounded = False
    stub.unsupported = ["Propranolol 40-360 mg/day in divided doses."]
    set_retrieval(monkeypatch, [chunk])

    state = await run()

    assert state["abstained"] is True


async def test_verifier_rejection_without_named_claims_still_rejects(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to check against the passages means nothing to overrule."""
    stub.grounded = False
    stub.unsupported = []
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["abstained"] is True

