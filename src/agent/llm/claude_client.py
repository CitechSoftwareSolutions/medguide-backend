"""The single place this application talks to the Claude API.

Every Claude Sonnet 5 request constraint is enforced here so that no graph node
can trip over one:

* ``temperature``, ``top_p`` and ``top_k`` are removed on Sonnet 5 and return
  HTTP 400. Determinism comes from deterministic retrieval and strict prompts,
  never from a sampling parameter.
* ``thinking={"type": "adaptive"}`` is the only on-mode; ``budget_tokens`` is
  rejected. ``{"type": "disabled"}`` is accepted and used for cheap graders.
* Assistant prefill is rejected, so response shape is constrained with
  ``output_config.format`` instead.
* Mid-conversation ``{"role": "system"}`` messages are unsupported on Sonnet 5,
  so all operator instruction stays in the top-level ``system`` field.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from src.config import get_settings
from src.exceptions import AgentUnavailableError

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

MessageParam = dict[str, Any]


@lru_cache
def get_client() -> anthropic.AsyncAnthropic:
    """Return the one shared asynchronous Claude client."""
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise AgentUnavailableError(
            "The assistant is not configured. Set ANTHROPIC_API_KEY to enable it."
        )
    return anthropic.AsyncAnthropic(
        api_key=settings.anthropic_api_key,
        max_retries=3,
        timeout=60.0,
    )


def _cached_system(system: str) -> list[dict[str, Any]]:
    """Wrap a stable system prompt in a cacheable block.

    The prompt must be byte-stable across requests for this to pay off; anything
    per-request (a timestamp, an identifier, the retrieved chunks) belongs in
    ``messages``, after this breakpoint.
    """
    return [
        {
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _log_cache_efficiency(label: str, usage: Any) -> None:
    """Record cache behaviour so a silent invalidator is visible in the logs."""
    read = getattr(usage, "cache_read_input_tokens", None)
    created = getattr(usage, "cache_creation_input_tokens", None)
    logger.debug(
        "claude call %s: cache_read=%s cache_created=%s input=%s output=%s",
        label,
        read,
        created,
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _translate(exc: Exception, label: str) -> AgentUnavailableError:
    """Map an SDK failure onto the application's error envelope.

    The chain is ordered most specific first; collapsing it into a single broad
    handler would erase the retryable versus non-retryable distinction.
    """
    request_id = getattr(exc, "request_id", None)
    if isinstance(exc, anthropic.NotFoundError):
        logger.error("Claude model or endpoint not found (%s) [%s]", label, request_id)
        message = "The assistant is misconfigured and cannot answer right now."
    elif isinstance(exc, anthropic.AuthenticationError):
        logger.error("Claude rejected the API key (%s) [%s]", label, request_id)
        message = "The assistant is not configured correctly."
    elif isinstance(exc, anthropic.RateLimitError):
        logger.warning("Claude rate limit hit (%s) [%s]", label, request_id)
        message = "The assistant is busy. Please retry in a few seconds."
    elif isinstance(exc, anthropic.APIStatusError):
        logger.error(
            "Claude returned status %s (%s) [%s]",
            exc.status_code,
            label,
            request_id,
        )
        message = "The assistant could not complete this request."
    elif isinstance(exc, anthropic.APIConnectionError):
        logger.error("Could not reach Claude (%s) [%s]", label, request_id)
        message = "The assistant is unreachable. Please retry shortly."
    else:
        logger.exception("Unexpected Claude failure (%s)", label, exc_info=exc)
        message = "The assistant could not complete this request."
    return AgentUnavailableError(message)


async def complete_text(
    *,
    system: str,
    messages: list[MessageParam],
    label: str,
    effort: str = "high",
    max_tokens: int | None = None,
) -> str:
    """Generate prose with adaptive thinking enabled.

    Uses the streaming endpoint even though the caller wants one string: a large
    ``max_tokens`` on a non-streaming request risks an HTTP timeout, and keeping
    the call streaming here means adding server-sent events later is an endpoint
    change rather than a change to this adapter.
    """
    settings = get_settings()
    client = get_client()
    try:
        async with client.messages.stream(
            model=settings.agent_model,
            max_tokens=max_tokens or settings.agent_max_answer_tokens,
            system=_cached_system(system),
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
        ) as stream:
            message = await stream.get_final_message()
    except Exception as exc:  # narrowed and re-raised by _translate
        raise _translate(exc, label) from exc

    _log_cache_efficiency(label, message.usage)

    if message.stop_reason == "refusal":
        logger.warning(
            "Claude refused during %s: %s",
            label,
            getattr(message.stop_details, "category", None),
        )
        raise AgentUnavailableError(
            "The assistant declined to answer this request."
        )

    return "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()


async def complete_structured(
    *,
    system: str,
    messages: list[MessageParam],
    schema: type[SchemaT],
    label: str,
    effort: str = "low",
    max_tokens: int = 2048,
) -> SchemaT:
    """Return a validated structured result from a cheap classification call.

    Thinking is disabled for these: routing, relevance grading and groundedness
    checking are classifications, not reasoning tasks, so they run at the lowest
    effort the API offers.
    """
    settings = get_settings()
    client = get_client()
    try:
        message = await client.messages.create(
            model=settings.agent_model,
            max_tokens=max_tokens,
            system=_cached_system(system),
            messages=messages,
            thinking={"type": "disabled"},
            output_config={
                "effort": effort,
                "format": {
                    "type": "json_schema",
                    "schema": schema.model_json_schema(),
                },
            },
        )
    except Exception as exc:  # narrowed and re-raised by _translate
        raise _translate(exc, label) from exc

    _log_cache_efficiency(label, message.usage)

    if message.stop_reason == "refusal":
        raise AgentUnavailableError("The assistant declined to answer this request.")

    payload = "".join(
        block.text for block in message.content if block.type == "text"
    ).strip()

    try:
        return schema.model_validate(json.loads(payload))
    except (json.JSONDecodeError, ValidationError) as exc:
        # Structured outputs make this near-impossible, but a malformed grader
        # result must surface as a handled failure rather than an AttributeError
        # deep inside a node.
        logger.error("Could not parse %s result: %s", label, payload[:500])
        raise AgentUnavailableError(
            "The assistant returned an unreadable result."
        ) from exc
