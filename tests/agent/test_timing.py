"""Tests for step timing: what each step reports and which model calls it owns."""

from __future__ import annotations

import asyncio

import pytest

from src.agent.timing import ModelCall, record_model_call, timed
from tests.graph_stubs import (
    StubModel,
    make_chunk,
    make_nice_chunk,
    run,
    set_nice_retrieval,
    set_retrieval,
)


def call(prompt_tokens: int, output_tokens: int) -> ModelCall:
    return ModelCall(
        prompt_tokens=prompt_tokens,
        prompt_seconds=prompt_tokens / 100,
        output_tokens=output_tokens,
        output_seconds=output_tokens / 10,
    )


async def test_a_step_without_model_calls_reports_only_its_duration() -> None:
    async def node(state):
        await asyncio.sleep(0.01)
        return {"answer": "x"}

    result = await timed("join_searches", node)({})

    [timing] = result["timings"]
    assert result["answer"] == "x"
    assert timing["step"] == "join_searches"
    assert timing["seconds"] >= 0.01
    assert "model_calls" not in timing


async def test_a_step_sums_the_model_calls_it_made() -> None:
    async def node(state):
        record_model_call(call(1000, 200))
        record_model_call(call(500, 100))
        return {}

    [timing] = (await timed("synthesize", node)({}))["timings"]

    assert timing["model_calls"] == 2
    assert timing["prompt_tokens"] == 1500
    assert timing["prompt_seconds"] == pytest.approx(15.0)
    assert timing["output_tokens"] == 300
    assert timing["output_seconds"] == pytest.approx(30.0)


async def test_steps_running_at_the_same_time_keep_their_own_calls() -> None:
    """Tool 01 and Tool 02 run together; neither may absorb the other's calls."""

    async def first(state):
        record_model_call(call(100, 10))
        await asyncio.sleep(0.01)
        return {}

    async def second(state):
        await asyncio.sleep(0.005)
        record_model_call(call(900, 90))
        return {}

    a, b = await asyncio.gather(timed("first", first)({}), timed("second", second)({}))

    assert a["timings"][0]["prompt_tokens"] == 100
    assert b["timings"][0]["prompt_tokens"] == 900


def test_a_model_call_outside_any_step_is_ignored() -> None:
    record_model_call(call(10, 1))  # must not raise


async def test_the_graph_reports_every_step_it_ran(
    stub: StubModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_retrieval(monkeypatch, [make_chunk()])
    set_nice_retrieval(monkeypatch, make_nice_chunk())

    state = await run()

    steps = [timing["step"] for timing in state["timings"]]
    assert steps[0] == "rewrite_query"
    assert set(steps[1:3]) == {"tool_01.retrieve_knowledge_base", "tool_02.retrieve_nice"}
    assert steps[3:] == [
        "join_searches",
        "synthesize",
        "verify_groundedness",
        "tool_02.validate_against_nice",
    ]
