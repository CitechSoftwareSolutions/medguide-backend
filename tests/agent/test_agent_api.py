"""Tests for the agent's entrypoint and its HTTP endpoint."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent import agent
from src.agent.agent import AgentAnswer
from src.agent.routes import router as agent_router
from src.agent.state import Citation
from src.shared.error_handlers import register_exception_handlers
from src.shared.errors import AgentUnavailableError, KnowledgeBaseEmptyError


def make_answer(**overrides) -> AgentAnswer:
    """Build one agent result for response-shaping tests."""
    fields = {
        "session_id": "session-abc",
        "answer": "Supportive care is indicated [1].",
        "citations": [
            Citation(
                marker=1,
                chunk_id="chunk-1",
                document_id="document-1",
                title="Migraine",
                score=0.82,
            )
        ],
        "confidence": "high",
        "route": "knowledge_qa",
    }
    return AgentAnswer(**{**fields, **overrides})


def client_answering(monkeypatch: pytest.MonkeyPatch, result) -> TestClient:
    """A client whose agent returns ``result``, or raises it if it is an exception."""

    async def answer_clinical_question(question, session_id=None):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(agent, "answer_clinical_question", answer_clinical_question)
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agent_router)
    return TestClient(app, raise_server_exceptions=False)


def test_ask_endpoint_returns_the_answer_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    with client_answering(monkeypatch, make_answer()) as client:
        response = client.post("/api/v1/agent/ask", json={"question": "How is migraine managed?"})

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["session_id"] == "session-abc"
    assert body["confidence"] == "high"
    assert body["citations"][0] == {
        "marker": 1,
        "chunk_id": "chunk-1",
        "document_id": "document-1",
        "title": "Migraine",
        "score": pytest.approx(0.82),
    }
    assert body["nice_validation"]["status"] == "not_run"


def test_ask_endpoint_returns_the_nice_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool 02's check travels to the client next to the answer."""
    answer = make_answer(
        nice_validation={
            "status": "conflicts",
            "summary": "NICE guidance disagrees with 1 claim in this answer.",
            "conflicts": [{"claim": "Give X.", "nice_position": "Give Y.", "markers": [1]}],
            "citations": [
                Citation(
                    marker=1,
                    chunk_id="nice-chunk-1",
                    document_id="nice-document",
                    title="Headaches",
                    score=0.7,
                )
            ],
        }
    )

    with client_answering(monkeypatch, answer) as client:
        body = client.post("/api/v1/agent/ask", json={"question": "Q?"}).json()["data"]

    validation = body["nice_validation"]
    assert validation["status"] == "conflicts"
    assert validation["conflicts"][0]["nice_position"] == "Give Y."
    assert validation["citations"][0]["title"] == "Headaches"


def test_ask_endpoint_rejects_a_blank_question() -> None:
    """Validation failures use the shared error envelope."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agent_router)

    with TestClient(app) as client:
        response = client.post("/api/v1/agent/ask", json={"question": "   "})

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_agent_outage_becomes_a_service_unavailable_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model outage surfaces as 503 in the standard envelope, not a 500."""
    outage = AgentUnavailableError("The assistant is unreachable.")

    with client_answering(monkeypatch, outage) as client:
        response = client.post("/api/v1/agent/ask", json={"question": "Question?"})

    assert response.status_code == 503
    assert response.json() == {
        "error": "agent_unavailable",
        "message": "The assistant is unreachable.",
    }


async def test_refuses_when_the_knowledge_base_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty Tool 01 store is a configuration problem with its own status code."""
    monkeypatch.setattr(agent.knowledge_base, "stats", lambda: {"document_count": 0, "chunk_count": 0})

    with pytest.raises(KnowledgeBaseEmptyError):
        await agent.answer_clinical_question("How is migraine managed?")


async def test_delegates_once_knowledge_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[tuple[str, object]] = []

    async def answer_question(question, session_id=None):
        received.append((question, session_id))
        return make_answer()

    monkeypatch.setattr(agent.knowledge_base, "stats", lambda: {"document_count": 1, "chunk_count": 3})
    monkeypatch.setattr(agent, "answer_question", answer_question)

    result = await agent.answer_clinical_question("Question?")

    assert received == [("Question?", None)]
    assert result.session_id == "session-abc"
