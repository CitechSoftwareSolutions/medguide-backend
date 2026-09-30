"""Tests for Tool 02's check inside the knowledge_qa graph.

Tool 02 searches NICE guidance alongside Tool 01 and checks the finished answer
against it. Every model call is stubbed; the tests cover when the check runs,
what it sees, and how its status is decided.
"""

from __future__ import annotations

import threading

import pytest

from src.agent.supervisor import nodes as supervisor_nodes
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_02_nice_guidance import service as tool_02
from tests.graph_stubs import (
    StubModel,
    make_chunk,
    make_nice_chunk,
    run,
    set_nice_retrieval,
    set_retrieval,
)

pytestmark = pytest.mark.usefixtures("no_nice_guidance")


async def test_both_tools_search_in_the_same_step(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each retriever waits for the other; run one after the other, both time out."""
    barrier = threading.Barrier(2, timeout=5)

    def knowledge(_query, top_k=None):
        barrier.wait()
        return [make_chunk()]

    def nice(_query, top_k=None):
        barrier.wait()
        return [make_nice_chunk()]

    monkeypatch.setattr(tool_01, "retrieve", knowledge)
    monkeypatch.setattr(tool_02, "retrieve", nice)

    state = await run()

    assert not state.get("abstained")
    assert state["nice_validation"]["status"] == "consistent"


async def test_answer_is_written_from_the_knowledge_base_only(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NICE text reaches the validator and nothing that writes or grades the answer."""
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk("NICE-ONLY-SENTINEL text."))

    await run()

    for label in ("grade_relevance", "synthesize", "verify_groundedness"):
        assert all("NICE-ONLY-SENTINEL" not in prompt for prompt in stub.prompts[label])
    assert "NICE-ONLY-SENTINEL" in stub.prompts["validate_against_nice"][0]


async def test_validator_sees_the_answer_without_knowledge_base_markers(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The answer's [n] markers point at passages the validator is not shown."""
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    await run()

    prompt = stub.prompts["validate_against_nice"][0]
    assert "Supportive care is indicated." in prompt
    assert "indicated [1]" not in prompt


async def test_tool_02_searches_with_the_rewritten_then_broadened_query(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both tools always search with the same query, including on a retry."""
    calls = {"count": 0}

    def knowledge(_query, top_k=None):
        calls["count"] += 1
        return [] if calls["count"] == 1 else [make_chunk()]

    monkeypatch.setattr(tool_01, "retrieve", knowledge)
    nice_queries = set_nice_retrieval(monkeypatch, make_nice_chunk())

    await run()

    assert nice_queries == ["rewrite_query-result", "broaden_query-result"]


async def test_consistent_answer_reports_nice_citations(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    validation = state["nice_validation"]
    assert validation["status"] == "consistent"
    assert validation["conflicts"] == []
    assert [citation["chunk_id"] for citation in validation["citations"]] == ["nice-chunk-1"]


async def test_conflict_is_reported_without_changing_the_answer(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clinician sees the knowledge base answer and the NICE disagreement together."""
    stub.nice_verdict = {
        "claims": [
            {
                "claim": "Supportive care is indicated.",
                "verdict": "contradicts",
                "nice_says": "Offer metformin first line.",
                "passages": [2, 9],
            }
        ]
    }
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk(number=1), make_nice_chunk(number=2))

    state = await run()

    assert state["answer"] == "Supportive care is indicated [1]."
    assert state["confidence"] == "high"
    validation = state["nice_validation"]
    assert validation["status"] == "conflicts"
    assert validation["summary"] == "NICE guidance disagrees with 1 claim in this answer."
    # Passage 9 does not exist and is dropped rather than trusted.
    assert validation["conflicts"] == [
        {
            "claim": "Supportive care is indicated.",
            "nice_position": "Offer metformin first line.",
            "markers": [2],
        }
    ]
    assert [citation["chunk_id"] for citation in validation["citations"]] == ["nice-chunk-2"]


def claim(verdict: str, nice_says: str = "", passages: list[int] | None = None) -> dict:
    return {
        "claim": f"A claim the model judged {verdict}.",
        "verdict": verdict,
        "nice_says": nice_says,
        "passages": passages or [],
    }


@pytest.mark.parametrize(
    ("claims", "expected_status"),
    [
        # One backed agreement makes the answer consistent despite unmentioned points.
        ([claim("agrees", "Same.", [1]), claim("not_addressed")], "consistent"),
        # Any backed contradiction is a conflict.
        ([claim("agrees", "Same.", [1]), claim("contradicts", "Other.", [1])], "conflicts"),
        # A "contradiction" that states no NICE position is a point NICE never made.
        ([claim("agrees", "Same.", [1]), claim("contradicts", "", [1])], "consistent"),
        # A "contradiction" that cites no real passage is discarded.
        ([claim("contradicts", "Other.", [9])], "not_covered"),
        # An agreement that cites no passage does not count as support.
        ([claim("agrees", "Same.", [])], "not_covered"),
        ([claim("not_addressed")], "not_covered"),
        ([], "not_covered"),
    ],
)
async def test_status_is_derived_from_backed_claim_verdicts(
    stub: StubModel,
    monkeypatch: pytest.MonkeyPatch,
    claims: list[dict],
    expected_status: str,
) -> None:
    """The model judges claims; the code decides the status and needs NICE evidence."""
    stub.nice_verdict = {"claims": claims}
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    assert state["nice_validation"]["status"] == expected_status


async def test_summary_counts_supported_and_unaddressed_claims(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub.nice_verdict = {
        "claims": [
            claim("agrees", "Same.", [1]),
            claim("not_addressed"),
            claim("contradicts", "", []),
        ]
    }
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    assert state["nice_validation"]["summary"] == (
        "NICE guidance supports 1 claim in this answer and contradicts none. "
        "2 claims not addressed by NICE."
    )


async def test_no_nice_passage_is_not_covered_without_a_model_call(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retrieval(monkeypatch, [make_chunk()])

    state = await run()

    assert state["nice_validation"]["status"] == "not_covered"
    assert "validate_against_nice" not in stub.calls


async def test_nice_search_failure_is_unavailable_and_the_answer_stands(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tool 02 fails open: its outage never costs the clinician the answer."""

    def broken(_query, top_k=None):
        raise RuntimeError("index corrupted")

    set_retrieval(monkeypatch, [make_chunk()])
    monkeypatch.setattr(tool_02, "retrieve", broken)

    state = await run()

    assert state["answer"] == "Supportive care is indicated [1]."
    assert state["nice_validation"]["status"] == "unavailable"
    assert "validate_against_nice" not in stub.calls


async def test_validator_outage_is_unavailable_and_the_answer_stands(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub.nice_raises = True
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    assert state["answer"] == "Supportive care is indicated [1]."
    assert state["nice_validation"]["status"] == "unavailable"


async def test_abstention_does_not_run_the_nice_check(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub.keep_passages = False
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    assert state["abstained"] is True
    assert state["nice_validation"]["status"] == "not_run"
    assert "validate_against_nice" not in stub.calls


async def test_out_of_scope_request_does_not_run_the_nice_check() -> None:
    result = await supervisor_nodes.decline_out_of_scope({})

    assert result["nice_validation"]["status"] == "not_run"
