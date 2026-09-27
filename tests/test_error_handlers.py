"""Tests for the global exception response format."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.dto import IndexDocumentRequest
from src.exceptions import DuplicateDocumentError, register_exception_handlers


def test_global_handlers_return_consistent_error_bodies() -> None:
    """Known, validation, and unexpected errors use the shared response format."""
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/duplicate")
    def duplicate_document() -> None:
        raise DuplicateDocumentError("Already indexed.")

    @app.post("/validate")
    def validate_document(payload: IndexDocumentRequest) -> IndexDocumentRequest:
        return payload

    @app.get("/unexpected")
    def unexpected_error() -> None:
        raise RuntimeError("Internal detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        duplicate = client.get("/duplicate")
        invalid = client.post("/validate", json={"title": "", "text": "x"})
        unexpected = client.get("/unexpected")

    assert duplicate.status_code == 409
    assert duplicate.json() == {"error": "duplicate_document", "message": "Already indexed."}
    assert invalid.status_code == 422
    assert invalid.json()["error"] == "validation_error"
    assert unexpected.status_code == 500
    assert unexpected.json() == {
        "error": "internal_server_error",
        "message": "An unexpected error occurred.",
    }
