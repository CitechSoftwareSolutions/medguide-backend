"""HTTP tests for the PDF upload endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.config.settings import get_settings
from src.exceptions import register_exception_handlers
from src.routes import rag_router

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")

MIGRAINE_PDF = (
    Path(__file__).resolve().parents[1] / "src" / "data" / "Migraine_Diagnosis_and_Management.pdf"
)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DOCUMENTS_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(rag_router)
    return TestClient(app)


def upload(client: TestClient, name: str, content: bytes):
    return client.post(
        "/api/v1/rag/documents/upload",
        files={"file": (name, content, "application/pdf")},
    )


def test_upload_indexes_the_pdf(client: TestClient, tmp_path: Path) -> None:
    response = upload(client, "Migraine.pdf", MIGRAINE_PDF.read_bytes())

    assert response.status_code == 201
    body = response.json()["data"]
    assert body["file_name"] == "Migraine.pdf"
    assert body["chunk_count"] > 0
    assert (tmp_path / "data" / "Migraine.pdf").exists()
    assert client.get("/api/v1/rag/stats").json()["data"]["document_count"] == 1


def test_duplicate_upload_returns_409(client: TestClient) -> None:
    upload(client, "Migraine.pdf", MIGRAINE_PDF.read_bytes())

    response = upload(client, "Copy.pdf", MIGRAINE_PDF.read_bytes())

    assert response.status_code == 409
    assert response.json()["error"] == "duplicate_document"


def test_same_name_upload_keeps_both(client: TestClient, tmp_path: Path) -> None:
    other = MIGRAINE_PDF.with_name("Diabetes_Diagnosis_and_Management.pdf")
    upload(client, "Guideline.pdf", MIGRAINE_PDF.read_bytes())

    response = upload(client, "Guideline.pdf", other.read_bytes())

    assert response.status_code == 201
    assert response.json()["data"]["file_name"] == "Guideline_2.pdf"
    assert sorted(path.name for path in (tmp_path / "data").iterdir()) == [
        "Guideline.pdf",
        "Guideline_2.pdf",
    ]
    assert client.get("/api/v1/rag/stats").json()["data"]["document_count"] == 2


def test_non_pdf_upload_returns_422(client: TestClient) -> None:
    response = upload(client, "notes.pdf", b"just text")

    assert response.status_code == 422
    assert response.json()["error"] == "invalid_document"


def test_oversized_upload_returns_413(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")
    get_settings.cache_clear()

    response = upload(client, "Migraine.pdf", MIGRAINE_PDF.read_bytes())

    assert response.status_code == 413
    assert response.json()["error"] == "document_too_large"


def test_missing_file_returns_422(client: TestClient) -> None:
    response = client.post("/api/v1/rag/documents/upload")

    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_list_documents_reports_each_document(client: TestClient) -> None:
    upload(client, "Migraine.pdf", MIGRAINE_PDF.read_bytes())
    client.post(
        "/api/v1/rag/documents",
        json={"title": "Asthma", "text": "Asthma is a chronic airway disease."},
    )

    response = client.get("/api/v1/rag/documents")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [item["title"] for item in body["data"]] == ["Asthma", "Migraine"]
    assert body["data"][0]["chunk_count"] == 1
    assert body["data"][1]["tags"] == ["pdf"]
