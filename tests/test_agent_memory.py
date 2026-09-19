"""Tests for session-scoped conversation memory.

Memory must persist across requests within one session identifier and must not
leak between different ones.
"""

from __future__ import annotations

import pytest

from src.agent import graph as graph_module
from src.agent.agent import answer_question
from src.agent.memory import checkpointer as checkpointer_module
from src.agent.memory import session as session_module
from src.agent.nodes import qa_nodes, supervisor_nodes
from src.agent.state import RetrievedChunk, append_turns
from src.agent.tools import registry
from src.config.settings import get_settings
from src.services import rag_service


@pytest.fixture
def stubbed_agent(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the real supervisor graph with every model call scripted."""
    checkpointer_module.get_checkpointer.cache_clear()
    graph_module.get_supervisor_graph.cache_clear()

    seen: dict[str, list[str]] = {"rewrite_prompts": []}

    async def supervisor_structured(*, schema, label, **_):
        assert label == "route"
        return schema(capability=registry.KNOWLEDGE_QA, reason="stub")

    async def qa_structured(*, schema, label, messages, **_):
        if label == "rewrite_query":
            seen["rewrite_prompts"].append(messages[0]["content"])
            return schema(search_query="stub query")
        if label == "grade_relevance":
            return schema(verdicts=[{"passage": 1, "keep": True, "reason": "stub"}])
        if label == "verify_groundedness":
            return schema(grounded=True, unsupported_claims=[])
        raise AssertionError(f"unexpected call: {label}")

    async def qa_text(*, label, **_):
        return "Supportive care is indicated [1]."

    monkeypatch.setattr(supervisor_nodes, "complete_structured", supervisor_structured)
    monkeypatch.setattr(qa_nodes, "complete_structured", qa_structured)
    monkeypatch.setattr(qa_nodes, "complete_text", qa_text)
    monkeypatch.setattr(
        rag_service,
        "retrieve",
        lambda _query, top_k=None: [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="document-1",
                title="Migraine",
                text="Migraine is managed with supportive care.",
                score=0.8,
            )
        ],
    )

    yield seen

    checkpointer_module.get_checkpointer.cache_clear()
    graph_module.get_supervisor_graph.cache_clear()


async def test_mints_a_session_id_when_none_is_given(stubbed_agent) -> None:
    """A first question starts a session the caller can continue."""
    result = await answer_question("How is migraine managed?")

    assert result.session_id
    assert result.route == registry.KNOWLEDGE_QA


async def test_follow_up_sees_the_earlier_exchange(stubbed_agent) -> None:
    """The second question is rewritten with the first turn in context."""
    first = await answer_question("How is migraine managed?")

    await answer_question("What about in children?", session_id=first.session_id)

    follow_up_prompt = stubbed_agent["rewrite_prompts"][1]
    assert "How is migraine managed?" in follow_up_prompt
    assert "Supportive care is indicated [1]." in follow_up_prompt
    assert "What about in children?" in follow_up_prompt


async def test_a_different_session_sees_no_earlier_history(stubbed_agent) -> None:
    """Sessions are isolated from one another."""
    await answer_question("How is migraine managed?")

    await answer_question("What about in children?")

    second_prompt = stubbed_agent["rewrite_prompts"][1]
    assert "How is migraine managed?" not in second_prompt
    assert "(no earlier conversation)" in second_prompt


async def test_history_grows_across_several_turns(stubbed_agent) -> None:
    """Each exchange adds to the same session's history."""
    first = await answer_question("First question?")
    await answer_question("Second question?", session_id=first.session_id)
    await answer_question("Third question?", session_id=first.session_id)

    third_prompt = stubbed_agent["rewrite_prompts"][2]
    assert "First question?" in third_prompt
    assert "Second question?" in third_prompt


def test_history_reducer_trims_to_the_configured_window() -> None:
    """The reducer bounds history so the checkpoint cannot grow forever."""
    max_turns = get_settings().session_max_turns
    existing = [{"role": "user", "content": f"turn {index}"} for index in range(max_turns)]

    combined = append_turns(existing, [{"role": "user", "content": "newest"}])

    assert len(combined) == max_turns
    assert combined[-1]["content"] == "newest"
    assert combined[0]["content"] == "turn 1", "the oldest turn is dropped"


def test_session_config_binds_the_thread_identifier() -> None:
    """The thread identifier is the whole mechanism behind cross-turn memory."""
    config = session_module.session_config("session-abc")

    assert config["configurable"]["thread_id"] == "session-abc"
    assert config["recursion_limit"] == get_settings().agent_recursion_limit


def test_idle_sessions_are_evicted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sweeping on write keeps the in-memory store bounded."""
    monkeypatch.setenv("SESSION_TTL_SECONDS", "0")
    get_settings.cache_clear()
    try:
        session_module.touch("first")
        session_module.touch("second")

        # With a zero lifetime every previously seen session is already idle.
        assert session_module.active_session_count() == 1
    finally:
        get_settings.cache_clear()


async def test_out_of_scope_requests_are_declined(
    stubbed_agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-medical request is refused without touching retrieval."""

    async def route_out_of_scope(*, schema, label, **_):
        return schema(capability=registry.OUT_OF_SCOPE, reason="stub")

    monkeypatch.setattr(supervisor_nodes, "complete_structured", route_out_of_scope)

    result = await answer_question("What is the weather today?")

    assert result.route == registry.OUT_OF_SCOPE
    assert result.confidence == "abstained"
    assert result.citations == []


async def test_unknown_route_falls_back_to_knowledge_lookup(
    stubbed_agent, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A router returning nonsense must not fail the request."""

    async def route_nonsense(*, schema, label, **_):
        return schema(capability="not_a_real_capability", reason="stub")

    monkeypatch.setattr(supervisor_nodes, "complete_structured", route_nonsense)

    result = await answer_question("How is migraine managed?")

    assert result.route == registry.KNOWLEDGE_QA
    assert result.answer == "Supportive care is indicated [1]."
