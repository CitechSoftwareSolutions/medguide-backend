"""Tests for the Claude adapter's guardrails.

These assert the request shape and the error translation without making a
network call. The request-shape test matters because Claude Sonnet 5 rejects
``temperature``, ``top_p``, ``top_k`` and ``thinking.budget_tokens`` with an
HTTP 400, and a regression there would only show up at runtime.
"""

from __future__ import annotations

import anthropic
import pytest

from src.agent.llm import claude_client
from src.config.settings import get_settings
from src.exceptions import AgentUnavailableError


class StubUsage:
    cache_read_input_tokens = 10
    cache_creation_input_tokens = 0
    input_tokens = 5
    output_tokens = 7


class StubBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class StubMessage:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [StubBlock(text)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = StubUsage()


class StubStream:
    def __init__(self, message: StubMessage) -> None:
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def get_final_message(self):
        return self._message


def install_stub_client(monkeypatch: pytest.MonkeyPatch, message: StubMessage) -> dict:
    """Replace the SDK client and capture the request keyword arguments."""
    captured: dict = {}

    class StubMessages:
        def stream(self, **kwargs):
            captured.update(kwargs)
            return StubStream(message)

        async def create(self, **kwargs):
            captured.update(kwargs)
            return message

    class StubClient:
        messages = StubMessages()

    monkeypatch.setattr(claude_client, "get_client", lambda: StubClient())
    return captured


async def test_text_request_omits_parameters_sonnet_five_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sampling parameters and a thinking budget must never be sent."""
    captured = install_stub_client(monkeypatch, StubMessage("An answer."))

    await claude_client.complete_text(
        system="You are a clinical assistant.",
        messages=[{"role": "user", "content": "Question?"}],
        label="test",
    )

    for rejected in ("temperature", "top_p", "top_k"):
        assert rejected not in captured
    assert captured["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in captured["thinking"]
    assert captured["model"] == get_settings().agent_model


async def test_text_request_marks_the_system_prompt_cacheable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The stable prefix carries a cache breakpoint; volatile text does not."""
    captured = install_stub_client(monkeypatch, StubMessage("An answer."))

    await claude_client.complete_text(
        system="Stable instructions.",
        messages=[{"role": "user", "content": "Volatile passages."}],
        label="test",
    )

    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["messages"][0]["content"] == "Volatile passages."


async def test_structured_request_disables_thinking_and_sets_a_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Classification calls run cheap, with a JSON schema constraining output."""
    from pydantic import BaseModel, ConfigDict

    class Verdict(BaseModel):
        model_config = ConfigDict(extra="forbid")

        keep: bool

    captured = install_stub_client(monkeypatch, StubMessage('{"keep": true}'))

    result = await claude_client.complete_structured(
        system="Judge this.",
        messages=[{"role": "user", "content": "Passage."}],
        schema=Verdict,
        label="test",
    )

    assert result.keep is True
    assert captured["thinking"] == {"type": "disabled"}
    assert captured["output_config"]["effort"] == "low"
    assert captured["output_config"]["format"]["type"] == "json_schema"
    assert captured["output_config"]["format"]["schema"]["additionalProperties"] is False


async def test_unparseable_structured_result_is_a_handled_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed output surfaces as a domain error, not an attribute error."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        keep: bool

    install_stub_client(monkeypatch, StubMessage("not json at all"))

    with pytest.raises(AgentUnavailableError):
        await claude_client.complete_structured(
            system="Judge this.",
            messages=[{"role": "user", "content": "Passage."}],
            schema=Verdict,
            label="test",
        )


async def test_refusal_is_reported_rather_than_returned_as_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal stop reason must not be mistaken for an answer."""
    install_stub_client(monkeypatch, StubMessage("", stop_reason="refusal"))

    with pytest.raises(AgentUnavailableError):
        await claude_client.complete_text(
            system="You are a clinical assistant.",
            messages=[{"role": "user", "content": "Question?"}],
            label="test",
        )


@pytest.mark.parametrize(
    "make_error",
    [
        # Built lazily: constructing SDK errors at collection time would make a
        # constructor change break collection rather than one test.
        lambda: anthropic.APIConnectionError(request=None),
        lambda: anthropic.APITimeoutError(request=None),
        lambda: RuntimeError("something unexpected"),
    ],
)
async def test_sdk_errors_become_agent_unavailable(
    monkeypatch: pytest.MonkeyPatch, make_error
) -> None:
    """Transport failures are translated into the application error envelope."""
    error = make_error()

    class StubMessages:
        def stream(self, **_):
            raise error

    class StubClient:
        messages = StubMessages()

    monkeypatch.setattr(claude_client, "get_client", lambda: StubClient())

    with pytest.raises(AgentUnavailableError):
        await claude_client.complete_text(
            system="You are a clinical assistant.",
            messages=[{"role": "user", "content": "Question?"}],
            label="test",
        )


def test_missing_api_key_is_reported_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unconfigured key fails with a usable message instead of a 401 later."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    get_settings.cache_clear()
    claude_client.get_client.cache_clear()
    try:
        with pytest.raises(AgentUnavailableError, match="not configured"):
            claude_client.get_client()
    finally:
        get_settings.cache_clear()
        claude_client.get_client.cache_clear()
