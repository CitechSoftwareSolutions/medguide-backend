"""Tests for session-scoped conversation memory.

Memory must persist across requests within one session identifier and must not
leak between different ones.
"""

from __future__ import annotations

import pytest

from src.agent import graph as graph_module
from src.agent.agent import answer_question
from src.agent.knowledge_qa import nodes as qa_nodes
from src.agent.memory import checkpointer as checkpointer_module
from src.agent.memory import session as session_module
from src.agent.state import RetrievedChunk, append_turns
from src.shared.config import get_settings
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_02_nice_guidance import service as tool_02


@pytest.fixture
def stubbed_agent(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Run the real session graph with every model call scripted."""
    checkpointer_module.get_checkpointer.cache_clear()
    graph_module.get_session_graph.cache_clear()

    seen: dict[str, list[str]] = {"rewrite_prompts": [], "queries": []}

    async def qa_structured(*, schema, label, messages, **_):
        if label == "rewrite_query":
            seen["rewrite_prompts"].append(messages[0]["content"])
            return schema(search_query="stub query")
        if label == "verify_groundedness":
            return schema(grounded=True, unsupported_claims=[])
        raise AssertionError(f"unexpected call: {label}")

    async def qa_text(*, label, **_):
        return "Supportive care is indicated [1]."

    def retrieve(query, top_k=None):
        seen["queries"].append(query)
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                document_id="document-1",
                title="Migraine",
                text="Migraine is managed with supportive care.",
                score=0.8,
            )
        ]

    monkeypatch.setattr(qa_nodes, "complete_structured", qa_structured)
    monkeypatch.setattr(qa_nodes, "complete_text", qa_text)
    # Tool 02 finds nothing, so no embedding model loads and no NICE call is made.
    monkeypatch.setattr(tool_02, "retrieve", lambda _query, top_k=None: [])
    monkeypatch.setattr(tool_01, "retrieve", retrieve)

    yield seen

    checkpointer_module.get_checkpointer.cache_clear()
    graph_module.get_session_graph.cache_clear()


async def test_mints_a_session_id_when_none_is_given(stubbed_agent) -> None:
    """A first question starts a session the caller can continue."""
    result = await answer_question("How is migraine managed?")

    assert result.session_id
    assert result.answer == "Supportive care is indicated [1]."


async def test_follow_up_sees_the_earlier_exchange(stubbed_agent) -> None:
    """The second question is rewritten with the first turn in context."""
    first = await answer_question("How is migraine managed?")

    await answer_question("What about in children?", session_id=first.session_id)

    [follow_up_prompt] = stubbed_agent["rewrite_prompts"]
    assert "How is migraine managed?" in follow_up_prompt
    assert "Supportive care is indicated [1]." in follow_up_prompt
    assert "What about in children?" in follow_up_prompt


async def test_a_different_session_sees_no_earlier_history(stubbed_agent) -> None:
    """Sessions are isolated: a new session's question is a first question."""
    await answer_question("How is migraine managed?")

    await answer_question("What about in children?")

    assert stubbed_agent["rewrite_prompts"] == []
    assert stubbed_agent["queries"] == ["How is migraine managed?", "What about in children?"]


async def test_history_grows_across_several_turns(stubbed_agent) -> None:
    """Each exchange adds to the same session's history."""
    first = await answer_question("First question?")
    await answer_question("Second question?", session_id=first.session_id)
    await answer_question("Third question?", session_id=first.session_id)

    third_prompt = stubbed_agent["rewrite_prompts"][1]
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
    # A clock that always advances: on Windows ``time.monotonic`` ticks about
    # every 15 ms, so two real calls in a row usually return the same value.
    # It starts after the real clock so sessions left by earlier tests count as idle.
    start = session_module.time.monotonic() + 1
    ticks = iter(range(1_000))
    monkeypatch.setattr(session_module.time, "monotonic", lambda: start + next(ticks))
    try:
        session_module.touch("first")
        session_module.touch("second")

        # With a zero lifetime every previously seen session is already idle.
        assert session_module.active_session_count() == 1
    finally:
        get_settings.cache_clear()

