"""The single place this application talks to the language model.

The model is MedGemma served by a local Ollama instance through its
``/api/chat`` endpoint. Constraints that matter here:

* ``temperature`` is 0. Answers must be repeatable for the same passages, and a
  4B model drifts noticeably at higher temperatures.
* ``repeat_penalty`` is raised to 1.2. MedGemma 1.5 opens every free-text reply
  with a thinking block that starts with the ``<unused94>`` token; under greedy
  decoding with Ollama's default penalty it emits that token forever.
* The thinking block, which ends at ``<unused95>``, is stripped from replies.
* ``num_ctx`` is set explicitly. Ollama's default context window is smaller
  than eight retrieved passages plus the system prompt, and it truncates the
  *start* of the prompt silently, which would drop the instructions. Every
  request, the warm-up included, must send the same value: a change makes
  Ollama reload the model.
* Structured calls pass the Pydantic JSON schema as ``format``. Ollama then
  constrains decoding to that schema, so the reply always parses; the schema
  is also written into the system prompt so a small model knows what each
  field means rather than only its shape.
"""

from __future__ import annotations

import json
import logging
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from src.shared.config import get_settings
from src.shared.errors import AgentUnavailableError

logger = logging.getLogger(__name__)

_THINKING_START = "<unused94>"
_THINKING_END = "<unused95>"

SchemaT = TypeVar("SchemaT", bound=BaseModel)

MessageParam = dict[str, Any]


def _timeout() -> httpx.Timeout:
    # Connecting is instant when Ollama is up; generation on a CPU can take
    # minutes, so only the read timeout is long.
    return httpx.Timeout(get_settings().ollama_timeout_seconds, connect=5.0)


def _options(max_tokens: int) -> dict[str, Any]:
    return {
        "temperature": 0,
        "repeat_penalty": 1.2,
        "num_ctx": get_settings().ollama_num_ctx,
        "num_predict": max_tokens,
    }


def _strip_thinking(content: str, label: str) -> str:
    """Return only the answer that follows MedGemma's thinking block.

    A block that never closed means the token limit ran out mid-thought, so
    there is no answer to return.
    """
    if _THINKING_END in content:
        return content.rsplit(_THINKING_END, 1)[1].strip()
    if content.lstrip().startswith(_THINKING_START):
        logger.warning("Model output for %s ended inside its thinking block", label)
        return ""
    return content.strip()


def _translate(exc: Exception, label: str) -> AgentUnavailableError:
    """Map a transport or server failure onto the application's error envelope."""
    settings = get_settings()
    if isinstance(exc, httpx.ConnectError):
        logger.error("Could not reach Ollama at %s (%s)", settings.ollama_base_url, label)
        message = "The assistant is unreachable. Is Ollama running?"
    elif isinstance(exc, httpx.TimeoutException):
        logger.error("Ollama timed out (%s)", label)
        message = "The assistant took too long to answer. Please retry."
    elif isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        logger.error("Ollama has no model named %s (%s)", settings.ollama_model, label)
        message = "The assistant is misconfigured and cannot answer right now."
    elif isinstance(exc, httpx.HTTPStatusError):
        logger.error(
            "Ollama returned status %s (%s): %s",
            exc.response.status_code,
            label,
            exc.response.text[:500],
        )
        message = "The assistant could not complete this request."
    else:
        logger.exception("Unexpected model failure (%s)", label, exc_info=exc)
        message = "The assistant could not complete this request."
    return AgentUnavailableError(message)


async def _chat(
    *,
    system: str,
    messages: list[MessageParam],
    label: str,
    max_tokens: int,
    response_format: dict[str, Any] | None = None,
) -> str:
    """Send one non-streaming chat request and return the reply text."""
    settings = get_settings()
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
        "keep_alive": settings.ollama_keep_alive,
        "options": _options(max_tokens),
    }
    if response_format is not None:
        payload["format"] = response_format

    try:
        async with httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=_timeout()
        ) as client:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            body = response.json()
    except Exception as exc:  # narrowed and re-raised by _translate
        raise _translate(exc, label) from exc

    logger.debug(
        "model call %s: prompt_tokens=%s output_tokens=%s done_reason=%s",
        label,
        body.get("prompt_eval_count"),
        body.get("eval_count"),
        body.get("done_reason"),
    )
    if body.get("done_reason") == "length":
        logger.warning("Model output for %s hit the %d-token limit", label, max_tokens)

    return _strip_thinking(str(body.get("message", {}).get("content", "")), label)


async def complete_text(
    *,
    system: str,
    messages: list[MessageParam],
    label: str,
    effort: str = "high",
    max_tokens: int | None = None,
) -> str:
    """Generate prose.

    ``effort`` is accepted for interface compatibility with the graph nodes and
    has no Ollama equivalent.
    """
    del effort
    answer = await _chat(
        system=system,
        messages=messages,
        label=label,
        max_tokens=max_tokens or get_settings().agent_max_answer_tokens,
    )
    if not answer:
        raise AgentUnavailableError("The assistant ran out of room before answering.")
    return answer


async def complete_structured(
    *,
    system: str,
    messages: list[MessageParam],
    schema: type[SchemaT],
    label: str,
    effort: str = "low",
    max_tokens: int = 2048,
) -> SchemaT:
    """Return a validated structured result from a classification call."""
    del effort
    json_schema = schema.model_json_schema()
    payload = await _chat(
        system=(
            f"{system}\n\nRespond with only a JSON object that matches this JSON "
            f"schema:\n{json.dumps(json_schema)}"
        ),
        messages=messages,
        label=label,
        max_tokens=max_tokens,
        response_format=json_schema,
    )

    try:
        return schema.model_validate(json.loads(payload))
    except (json.JSONDecodeError, ValidationError) as exc:
        # Constrained decoding makes this rare, but output cut off by the token
        # limit is still invalid JSON and must surface as a handled failure.
        logger.error("Could not parse %s result: %s", label, payload[:500])
        raise AgentUnavailableError(
            "The assistant returned an unreadable result."
        ) from exc


def warm_chat_model() -> None:
    """Load the model into Ollama's memory so the first question is not slow.

    A generate request with no prompt only loads the model. Failure is logged,
    not raised: Ollama may simply not be started yet.
    """
    settings = get_settings()
    try:
        httpx.post(
            f"{settings.ollama_base_url}/api/generate",
            json={
                "model": settings.ollama_model,
                "keep_alive": settings.ollama_keep_alive,
                "options": {"num_ctx": settings.ollama_num_ctx},
            },
            timeout=_timeout(),
        ).raise_for_status()
        logger.info("Loaded %s in Ollama", settings.ollama_model)
    except httpx.HTTPError as exc:
        logger.warning("Could not preload %s in Ollama: %s", settings.ollama_model, exc)
