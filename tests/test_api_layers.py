"""Tests for controllers, routes, and global exception response formats."""

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.controllers import knowledge_controller
from src.dto import CreateKnowledgeEntryRequest, KnowledgeEntryResponse
from src.enums import KnowledgeType
from src.exceptions import KnowledgeEntryNotFoundError, register_exception_handlers
from src.models import KnowledgeEntry
from src.routes import knowledge_routes


def make_entry() -> KnowledgeEntry:
    """Create one ORM model for response-shaping tests."""
    return KnowledgeEntry(
        id=uuid4(),
        title="Migraine",
        summary="A recurrent headache disorder.",
        knowledge_type=KnowledgeType.CONDITION,
        tags=["neurology"],
        created_at=datetime.now(UTC),
    )


def test_controller_serializes_orm_entry(monkeypatch) -> None:
    """Controllers convert ORM records to public Pydantic response DTOs."""
    entry = make_entry()
    monkeypatch.setattr(
        knowledge_controller.knowledge_service,
        "get_knowledge_entry",
        lambda _: entry,
    )

    response = knowledge_controller.get_entry(entry.id)

    assert response.data.id == entry.id
    assert response.data.knowledge_type == KnowledgeType.CONDITION


def test_route_forwards_payload_to_controller(monkeypatch) -> None:
    """Routes remain thin HTTP adapters."""
    payload = CreateKnowledgeEntryRequest(
        title="Migraine",
        summary="A recurrent headache disorder.",
        knowledge_type=KnowledgeType.CONDITION,
    )
    response = KnowledgeEntryResponse.model_validate({"data": make_entry()}, from_attributes=True)
    received_payloads: list[CreateKnowledgeEntryRequest] = []
    monkeypatch.setattr(
        knowledge_routes,
        "create_entry",
        lambda received_payload: received_payloads.append(received_payload) or response,
    )

    assert knowledge_routes.create_knowledge_entry(payload) == response
    assert received_payloads == [payload]


def test_global_handlers_return_consistent_error_bodies() -> None:
    """Known, validation, and unexpected errors use the shared response format."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/missing")
    def missing_entry() -> None:
        raise KnowledgeEntryNotFoundError("Entry does not exist.")

    @app.post("/validate")
    def validate_entry(payload: CreateKnowledgeEntryRequest) -> CreateKnowledgeEntryRequest:
        return payload

    @app.get("/unexpected")
    def unexpected_error() -> None:
        raise RuntimeError("Internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get("/missing")
        invalid = client.post("/validate", json={"title": "", "summary": "x", "knowledge_type": "condition"})
        unexpected = client.get("/unexpected")

    assert missing.status_code == 404
    assert missing.json()["error"] == "knowledge_entry_not_found"
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"
    assert unexpected.status_code == 500
    assert unexpected.json() == {
        "error": "internal_server_error",
        "message": "An unexpected error occurred.",
    }
