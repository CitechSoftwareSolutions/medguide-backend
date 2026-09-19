"""Tests for the agent and indexing controllers, routes and error bodies."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.agent.agent import AgentAnswer
from src.agent.state import Citation
from src.controllers import agent_controller, rag_controller
from src.dto import IndexDocumentRequest
from src.exceptions import (
    AgentUnavailableError,
    KnowledgeBaseEmptyError,
    register_exception_handlers,
)
from src.routes import agent_router, rag_router
from src.services import agent_service, rag_service


def make_answer() -> AgentAnswer:
    """Build one agent result for response-shaping tests."""
    return AgentAnswer(
        session_id="session-abc",
        answer="Supportive care is indicated [1].",
        citations=[
            Citation(
                marker=1,
                chunk_id="chunk-1",
                document_id="document-1",
                title="Migraine",
                score=0.82,
            )
        ],
        confidence="high",
        route="knowledge_qa",
    )


async def test_controller_shapes_the_agent_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Controllers convert agent results into public response DTOs."""
    answer = make_answer()

    async def answer_clinical_question(question, session_id=None):
        return answer

    monkeypatch.setattr(
        agent_controller.agent_service,
        "answer_clinical_question",
        answer_clinical_question,
    )

    response = await agent_controller.ask("How is migraine managed?")

    assert response.data.session_id == "session-abc"
    assert response.data.confidence == "high"
    assert response.data.citations[0].chunk_id == "chunk-1"
    assert response.data.citations[0].score == pytest.approx(0.82)


async def test_service_refuses_when_nothing_is_indexed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty index is a configuration problem with its own status code."""
    monkeypatch.setattr(
        agent_service.rag_service,
        "stats",
        lambda: {"document_count": 0, "chunk_count": 0},
    )

    with pytest.raises(KnowledgeBaseEmptyError):
        await agent_service.answer_clinical_question("How is migraine managed?")


async def test_service_delegates_once_knowledge_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With knowledge indexed the service forwards to the agent."""
    received: list[tuple[str, object]] = []

    async def answer_question(question, session_id=None):
        received.append((question, session_id))
        return make_answer()

    monkeypatch.setattr(
        agent_service.rag_service,
        "stats",
        lambda: {"document_count": 1, "chunk_count": 3},
    )
    monkeypatch.setattr(agent_service, "answer_question", answer_question)

    result = await agent_service.answer_clinical_question("Question?")

    assert received == [("Question?", None)]
    assert result.session_id == "session-abc"


def test_rag_controller_shapes_indexing_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """The indexing controller returns the identifier needed for deletion."""
    monkeypatch.setattr(
        rag_controller.rag_service,
        "index_document",
        lambda title, text, tags: {"document_id": "document-1", "chunk_count": 2},
    )

    response = rag_controller.index_document(
        IndexDocumentRequest(title="Migraine", text="A headache disorder.")
    )

    assert response.data.document_id == "document-1"
    assert response.data.chunk_count == 2


def test_ask_endpoint_returns_the_answer_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The route is a thin adapter over the controller."""
    answer = make_answer()

    async def answer_clinical_question(question, session_id=None):
        return answer

    monkeypatch.setattr(
        agent_service, "answer_clinical_question", answer_clinical_question
    )
    monkeypatch.setattr(
        agent_controller.agent_service,
        "answer_clinical_question",
        answer_clinical_question,
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agent_router)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/agent/ask", json={"question": "How is migraine managed?"}
        )

    assert response.status_code == 200
    body = response.json()["data"]
    assert body["session_id"] == "session-abc"
    assert body["citations"][0]["marker"] == 1


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

    async def answer_clinical_question(question, session_id=None):
        raise AgentUnavailableError("The assistant is unreachable.")

    monkeypatch.setattr(
        agent_controller.agent_service,
        "answer_clinical_question",
        answer_clinical_question,
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(agent_router)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/agent/ask", json={"question": "Question?"})

    assert response.status_code == 503
    assert response.json() == {
        "error": "agent_unavailable",
        "message": "The assistant is unreachable.",
    }


def test_stats_endpoint_reports_index_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """The statistics endpoint exposes what is currently searchable."""
    monkeypatch.setattr(
        rag_service, "stats", lambda: {"document_count": 4, "chunk_count": 11}
    )
    monkeypatch.setattr(
        rag_controller.rag_service,
        "stats",
        lambda: {"document_count": 4, "chunk_count": 11},
    )

    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(rag_router)

    with TestClient(app) as client:
        response = client.get("/api/v1/rag/stats")

    assert response.status_code == 200
    assert response.json()["data"] == {"document_count": 4, "chunk_count": 11}
