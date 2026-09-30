"""Tests for Tool 02: its NICE store, PDF handling and endpoints under /api/v1/nice."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared import documents
from src.shared.config import get_settings
from src.shared.error_handlers import register_exception_handlers
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_01_knowledge_base.routes import router as rag_router
from src.tools.tool_02_nice_guidance import service as tool_02
from src.tools.tool_02_nice_guidance.routes import router as nice_router
from tests.sample_pdfs import HEADACHES_PDF, TYPE_2_DIABETES_PDF

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")


def test_each_tool_searches_only_its_own_store() -> None:
    """A NICE passage can never become answer material, and vice versa."""
    tool_01.index_document("Knowledge migraine", "Migraine responds to aspirin.")
    tool_02.index_document("NICE migraine", "Offer aspirin for acute migraine.")

    knowledge_titles = {chunk["title"] for chunk in tool_01.retrieve("migraine aspirin")}
    nice_titles = {chunk["title"] for chunk in tool_02.retrieve("migraine aspirin")}

    assert knowledge_titles == {"Knowledge migraine"}
    assert nice_titles == {"NICE migraine"}


def test_nice_store_persists_to_its_own_directory() -> None:
    tool_02.index_document("NICE asthma", "Asthma guidance.")

    settings = get_settings()
    assert (settings.nice_vector_store_dir / "index.faiss").exists()
    assert settings.nice_vector_store_dir != settings.knowledge_base_vector_store_dir
    assert tool_01.stats()["chunk_count"] == 0


def test_nice_titles_come_from_pdf_metadata(tmp_path: Path) -> None:
    """NICE file names carry download ids; the metadata carries the real title."""
    shutil.copy(TYPE_2_DIABETES_PDF, tmp_path)

    tool_02.sync(tmp_path)

    [document] = tool_02.list_documents()
    assert document["title"] == "Type 2 diabetes in adults: management"


def test_extraction_drops_nice_footers_and_contents_leaders() -> None:
    """The rights notice wraps two ways; the repeated-line filter catches only one."""
    text = documents.extract_pdf_text(TYPE_2_DIABETES_PDF)

    assert "Notice of rights" not in text
    assert "....." not in text
    assert "Page 21 of" not in text
    assert "metformin" in text


def test_rebuilding_nice_leaves_the_knowledge_base_alone(tmp_path: Path) -> None:
    tool_01.index_document("Knowledge migraine", "Migraine responds to aspirin.")
    shutil.copy(HEADACHES_PDF, tmp_path)

    stats = tool_02.rebuild(tmp_path)

    assert stats["document_count"] == 1
    assert tool_01.stats()["document_count"] == 1


def test_nice_retrieval_drops_repeated_passages() -> None:
    """NICE repeats paragraphs across sections; the check sees each once."""
    for _ in range(3):
        tool_02.index_document("NICE migraine", "Offer aspirin for acute migraine.")
    tool_02.index_document("NICE migraine", "Migraine aspirin dosing guidance.")

    texts = [chunk["text"] for chunk in tool_02.retrieve("migraine aspirin", top_k=2)]

    assert len(texts) == 2
    assert len(set(texts)) == 2


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("KNOWLEDGE_BASE_DOCUMENTS_DIR", str(tmp_path / "tool_01"))
    monkeypatch.setenv("NICE_DOCUMENTS_DIR", str(tmp_path / "tool_02"))
    get_settings.cache_clear()
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(rag_router)
    app.include_router(nice_router)
    return TestClient(app)


def test_nice_upload_saves_into_the_tool_02_folder(client: TestClient, tmp_path: Path) -> None:
    response = client.post(
        "/api/v1/nice/documents/upload",
        files={"file": ("headaches.pdf", HEADACHES_PDF.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 201
    assert response.json()["data"]["title"] == "Headaches"
    assert (tmp_path / "tool_02" / "headaches.pdf").exists()
    assert client.get("/api/v1/nice/stats").json()["data"]["document_count"] == 1
    assert client.get("/api/v1/rag/stats").json()["data"]["document_count"] == 0


def test_nice_duplicate_upload_names_the_nice_corpus(client: TestClient) -> None:
    content = HEADACHES_PDF.read_bytes()
    client.post("/api/v1/nice/documents/upload", files={"file": ("a.pdf", content)})

    response = client.post("/api/v1/nice/documents/upload", files={"file": ("b.pdf", content)})

    assert response.status_code == 409
    assert "NICE guidance" in response.json()["message"]


def test_nice_text_indexing_listing_and_removal(client: TestClient) -> None:
    created = client.post(
        "/api/v1/nice/documents",
        json={"title": "NICE asthma", "text": "Asthma guidance for adults."},
    )
    assert created.status_code == 201
    document_id = created.json()["data"]["document_id"]

    listed = client.get("/api/v1/nice/documents").json()
    assert [doc["title"] for doc in listed["data"]] == ["NICE asthma"]

    removed = client.delete(f"/api/v1/nice/documents/{document_id}")
    assert removed.json()["data"] == {"document_count": 0, "chunk_count": 0}
