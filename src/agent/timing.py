"""How long each step of the knowledge_qa graph took, and where the time went.

Every node is wrapped with :func:`timed` when the graph is built. The wrapper
measures the step and adds one :class:`StepTiming` to the ``timings`` state
list, which is returned with the answer and logged as it happens::

    step tool_01.retrieve_knowledge_base: 0.05 s
    step synthesize: 38.2 s (model: 1850 prompt tokens in 18.6 s, 310 output tokens in 19.3 s)

For steps that call MedGemma, the Ollama client reports each call's token
counts and durations through :func:`record_model_call`. A context variable
ties those reports to the step that made them, including when Tool 01 and
Tool 02 run at the same time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import Any, NotRequired, TypedDict

logger = logging.getLogger(__name__)


class ModelCall(TypedDict):
    """What Ollama reported for one call."""

    prompt_tokens: int
    prompt_seconds: float
    output_tokens: int
    output_seconds: float


class StepTiming(TypedDict):
    """One step of the graph, in the order it finished."""

    step: str
    seconds: float
    # Present only for steps that called the model; summed over their calls.
    model_calls: NotRequired[int]
    prompt_tokens: NotRequired[int]
    prompt_seconds: NotRequired[float]
    output_tokens: NotRequired[int]
    output_seconds: NotRequired[float]


_current_calls: ContextVar[list[ModelCall] | None] = ContextVar("model_calls", default=None)


def record_model_call(call: ModelCall) -> None:
    """Attribute one model call to the step that is running, if any."""
    calls = _current_calls.get()
    if calls is not None:
        calls.append(call)


def _describe(timing: StepTiming) -> str:
    text = f"step {timing['step']}: {timing['seconds']:.2f} s"
    if "model_calls" in timing:
        text += (
            f" (model: {timing['prompt_tokens']} prompt tokens in "
            f"{timing['prompt_seconds']:.1f} s, {timing['output_tokens']} output tokens "
            f"in {timing['output_seconds']:.1f} s)"
        )
    return text


Node = Callable[[Any], Awaitable[dict]]


def timed(step: str, node: Node) -> Node:
    """Wrap a graph node so it reports how long it took."""

    async def run(state: Any) -> dict:
        calls: list[ModelCall] = []
        token = _current_calls.set(calls)
        started = time.perf_counter()
        try:
            result = await node(state)
        finally:
            _current_calls.reset(token)
        timing = StepTiming(step=step, seconds=round(time.perf_counter() - started, 3))
        if calls:
            timing["model_calls"] = len(calls)
            timing["prompt_tokens"] = sum(call["prompt_tokens"] for call in calls)
            timing["prompt_seconds"] = round(sum(call["prompt_seconds"] for call in calls), 3)
            timing["output_tokens"] = sum(call["output_tokens"] for call in calls)
            timing["output_seconds"] = round(sum(call["output_seconds"] for call in calls), 3)
        logger.info(_describe(timing))
        return {**result, "timings": [timing]}

    run.__name__ = getattr(node, "__name__", step)
    return run


def log_summary(timings: list[StepTiming], total_seconds: float) -> None:
    """Log the whole question's time and how much of it the model took."""
    model_seconds = sum(
        timing.get("prompt_seconds", 0.0) + timing.get("output_seconds", 0.0)
        for timing in timings
    )
    model_calls = sum(timing.get("model_calls", 0) for timing in timings)
    logger.info(
        "question answered in %.1f s: %d model call(s) took %.1f s, %d step(s)",
        total_seconds,
        model_calls,
        model_seconds,
        len(timings),
    )
