"""Tests for the Ollama adapter.

These assert the request shape and the error translation against a mocked HTTP
transport, so no Ollama server is needed.
"""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from src.agent import ollama_client
from src.shared.config import get_settings
from src.shared.errors import AgentUnavailableError


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep: bool


def install_transport(monkeypatch: pytest.MonkeyPatch, handler) -> list[dict]:
    """Route the adapter's HTTP calls to ``handler`` and capture request bodies."""
    captured: list[dict] = []
    real_client = httpx.AsyncClient

    def recording_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return handler(request)

    monkeypatch.setattr(
        ollama_client.httpx,
        "AsyncClient",
        lambda **kwargs: real_client(transport=httpx.MockTransport(recording_handler), **kwargs),
    )
    return captured


def chat_reply(content: str, done_reason: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": content},
            "done_reason": done_reason,
            "prompt_eval_count": 12,
            "eval_count": 5,
        },
    )


async def test_text_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deterministic sampling, an explicit context window, and a system message."""
    captured = install_transport(monkeypatch, lambda _: chat_reply("  An answer.  "))

    answer = await ollama_client.complete_text(
        system="You are a clinical assistant.",
        messages=[{"role": "user", "content": "Question?"}],
        label="test",
    )

    settings = get_settings()
    request = captured[0]
    assert answer == "An answer."
    assert request["model"] == settings.ollama_model
    assert request["stream"] is False
    assert request["messages"][0] == {
        "role": "system",
        "content": "You are a clinical assistant.",
    }
    assert request["messages"][1]["content"] == "Question?"
    assert request["options"] == {
        "temperature": 0,
        "repeat_penalty": 1.2,
        "num_ctx": settings.ollama_num_ctx,
        "num_predict": settings.agent_max_answer_tokens,
    }
    assert "format" not in request


async def test_thinking_block_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the text after MedGemma's thinking block is the answer."""
    install_transport(
        monkeypatch,
        lambda _: chat_reply("<unused94>thought\nReason it through.<unused95>Answer [1]."),
    )

    answer = await ollama_client.complete_text(
        system="System.", messages=[{"role": "user", "content": "Hi"}], label="test"
    )

    assert answer == "Answer [1]."


async def test_unfinished_thinking_is_a_handled_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running out of tokens mid-thought must not return the reasoning as the answer."""
    install_transport(
        monkeypatch, lambda _: chat_reply("<unused94>thought\nStill going", "length")
    )

    with pytest.raises(AgentUnavailableError, match="ran out of room"):
        await ollama_client.complete_text(
            system="System.", messages=[{"role": "user", "content": "Hi"}], label="test"
        )


async def test_structured_request_constrains_output_to_the_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The schema is sent as ``format`` and described in the system prompt."""
    captured = install_transport(monkeypatch, lambda _: chat_reply('{"keep": true}'))

    result = await ollama_client.complete_structured(
        system="Judge this.",
        messages=[{"role": "user", "content": "Passage."}],
        schema=Verdict,
        label="test",
    )

    assert result.keep is True
    assert captured[0]["format"] == Verdict.model_json_schema()
    assert '"keep"' in captured[0]["messages"][0]["content"]


async def test_unparseable_structured_output_is_a_handled_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON cut off at the token limit must not escape as a raw exception."""
    install_transport(monkeypatch, lambda _: chat_reply('{"keep": tr', "length"))

    with pytest.raises(AgentUnavailableError):
        await ollama_client.complete_structured(
            system="Judge this.",
            messages=[{"role": "user", "content": "Passage."}],
            schema=Verdict,
            label="test",
        )


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (httpx.ConnectError("refused"), "Is Ollama running?"),
        (httpx.ReadTimeout("slow"), "took too long"),
    ],
)
async def test_transport_failures_become_agent_unavailable(
    monkeypatch: pytest.MonkeyPatch, failure: Exception, message: str
) -> None:
    def fail(_: httpx.Request) -> httpx.Response:
        raise failure

    install_transport(monkeypatch, fail)

    with pytest.raises(AgentUnavailableError, match=message):
        await ollama_client.complete_text(
            system="System.", messages=[{"role": "user", "content": "Hi"}], label="test"
        )


async def test_missing_model_becomes_agent_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_transport(
        monkeypatch, lambda _: httpx.Response(404, json={"error": "model not found"})
    )

    with pytest.raises(AgentUnavailableError, match="misconfigured"):
        await ollama_client.complete_text(
            system="System.", messages=[{"role": "user", "content": "Hi"}], label="test"
        )


async def test_each_call_reports_its_tokens_and_durations_to_the_running_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ollama's nanosecond durations are converted and handed to the step timer."""
    from src.agent.timing import timed

    reply = httpx.Response(
        200,
        json={
            "message": {"role": "assistant", "content": "An answer."},
            "done_reason": "stop",
            "prompt_eval_count": 1800,
            "prompt_eval_duration": 18_000_000_000,
            "eval_count": 240,
            "eval_duration": 30_000_000_000,
        },
    )
    install_transport(monkeypatch, lambda _: reply)

    async def node(state):
        await ollama_client.complete_text(
            system="s", messages=[{"role": "user", "content": "q"}], label="synthesize"
        )
        return {}

    [timing] = (await timed("synthesize", node)({}))["timings"]

    assert timing["model_calls"] == 1
    assert timing["prompt_tokens"] == 1800
    assert timing["prompt_seconds"] == pytest.approx(18.0)
    assert timing["output_tokens"] == 240
    assert timing["output_seconds"] == pytest.approx(30.0)
