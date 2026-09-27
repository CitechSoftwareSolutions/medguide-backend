"""Tests for loading the PDF guidelines into the vector store."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.exceptions import DuplicateDocumentError, InvalidDocumentError
from src.repositories import vector_repository
from src.services import document_service, rag_service

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")

DATA_DIR = Path(__file__).resolve().parents[1] / "src" / "data"
MIGRAINE_PDF = DATA_DIR / "Migraine_Diagnosis_and_Management.pdf"
TREMOR_PDF = DATA_DIR / "Chronic_Hand_Tremor_Diagnosis_and_Management.pdf"
DIABETES_PDF = DATA_DIR / "Diabetes_Diagnosis_and_Management.pdf"


def test_extraction_drops_running_headers_and_page_numbers() -> None:
    """Page furniture would otherwise be embedded into every chunk."""
    text = document_service.extract_pdf_text(MIGRAINE_PDF)

    assert "ICHD-3" in text
    assert "Page 1 of" not in text
    assert text.count("Clinical Teaching Note") <= 1


def test_title_comes_from_the_file_name() -> None:
    """Underscores, the .docx remnant and a copy suffix are cleaned away."""
    path = Path("Dizziness_in_Elderly_Assessment_and_Management_1.docx.pdf")

    assert document_service.document_title(path) == (
        "Dizziness in Elderly Assessment and Management"
    )


def test_sync_indexes_each_pdf_once(tmp_path: Path) -> None:
    """A second sync over unchanged files embeds nothing."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    shutil.copy(TREMOR_PDF, tmp_path)

    first = document_service.sync_documents_directory(tmp_path)
    chunks_after_first = vector_repository.stats()["chunk_count"]
    second = document_service.sync_documents_directory(tmp_path)

    assert first == {"added": 2, "removed": 0}
    assert second == {"added": 0, "removed": 0}
    assert vector_repository.stats() == {
        "document_count": 2,
        "chunk_count": chunks_after_first,
    }


def test_sync_indexes_every_pdf_that_shares_a_title(tmp_path: Path) -> None:
    """Documents are told apart by content, so same-titled PDFs all go in."""
    shutil.copy(MIGRAINE_PDF, tmp_path / "Guideline.pdf")
    shutil.copy(TREMOR_PDF, tmp_path / "Guideline_1.pdf")
    shutil.copy(DIABETES_PDF, tmp_path / "Guideline 2.pdf")

    result = document_service.sync_documents_directory(tmp_path)

    assert result == {"added": 3, "removed": 0}
    assert vector_repository.stats()["document_count"] == 3


def test_title_keeps_a_year_suffix() -> None:
    """Only short copy suffixes are stripped from the title."""
    assert document_service.document_title(Path("Diabetes_2024.pdf")) == "Diabetes 2024"


def test_sync_drops_removed_pdfs_but_keeps_api_documents(tmp_path: Path) -> None:
    """Only chunks that came from the documents directory are pruned."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    document_service.sync_documents_directory(tmp_path)
    api_document = rag_service.index_document("Asthma", "Asthma is a chronic disease.")

    (tmp_path / MIGRAINE_PDF.name).unlink()
    result = document_service.sync_documents_directory(tmp_path)

    assert result == {"added": 0, "removed": 1}
    assert vector_repository.document_ids() == {api_document["document_id"]}


def test_indexed_pdf_is_retrievable(tmp_path: Path) -> None:
    """A question about the PDF's subject finds its chunks."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    document_service.sync_documents_directory(tmp_path)

    results = rag_service.retrieve("migraine aura")

    assert results
    assert results[0]["title"] == "Migraine Diagnosis and Management"


@pytest.fixture
def documents(tmp_path: Path) -> Path:
    """A documents directory apart from the vector store the fixture creates."""
    return tmp_path / "docs"


def test_upload_saves_and_indexes_the_pdf(documents: Path) -> None:
    """An upload is searchable immediately and lands in the documents directory."""
    result = document_service.upload_pdf(
        "Migraine_Diagnosis_and_Management.pdf", MIGRAINE_PDF.read_bytes(), documents
    )

    assert (documents / "Migraine_Diagnosis_and_Management.pdf").exists()
    assert result["title"] == "Migraine Diagnosis and Management"
    assert result["chunk_count"] > 0
    assert vector_repository.document_ids() == {result["document_id"]}
    # The startup sync must treat the uploaded file as already indexed.
    assert document_service.sync_documents_directory(documents) == {"added": 0, "removed": 0}


def test_upload_rejects_the_same_pdf_twice(documents: Path) -> None:
    """Identical content under any name is a duplicate."""
    content = MIGRAINE_PDF.read_bytes()
    document_service.upload_pdf("migraine.pdf", content, documents)

    with pytest.raises(DuplicateDocumentError):
        document_service.upload_pdf("another-name.pdf", content, documents)


def test_upload_with_an_existing_name_keeps_both_documents(documents: Path) -> None:
    """Nothing is replaced: the second PDF is saved alongside under a suffix."""
    first = document_service.upload_pdf("guideline.pdf", MIGRAINE_PDF.read_bytes(), documents)
    second = document_service.upload_pdf("guideline.pdf", TREMOR_PDF.read_bytes(), documents)
    third = document_service.upload_pdf("guideline.pdf", DIABETES_PDF.read_bytes(), documents)

    assert [first["file_name"], second["file_name"], third["file_name"]] == [
        "guideline.pdf",
        "guideline_2.pdf",
        "guideline_3.pdf",
    ]
    assert first["title"] == second["title"] == third["title"] == "guideline"
    assert vector_repository.document_ids() == {
        first["document_id"],
        second["document_id"],
        third["document_id"],
    }
    # A restart finds all three already indexed and removes none of them.
    assert document_service.sync_documents_directory(documents) == {
        "added": 0,
        "removed": 0,
    }


@pytest.mark.parametrize(
    ("file_name", "saved_as"),
    [
        ("../../main.py", "main.py.pdf"),
        ("..\..\evil.pdf", "evil.pdf"),
        ("Hand tremor (v2).pdf", "Hand tremor _v2_.pdf"),
    ],
)
def test_upload_file_names_cannot_escape_the_directory(
    documents: Path, file_name: str, saved_as: str
) -> None:
    """Directory parts are dropped and unusual characters replaced."""
    result = document_service.upload_pdf(file_name, MIGRAINE_PDF.read_bytes(), documents)

    assert result["file_name"] == saved_as
    assert [path.name for path in documents.iterdir()] == [saved_as]


@pytest.mark.parametrize(
    "content", [b"plain text, not a pdf", b"%PDF-1.7 truncated garbage"]
)
def test_upload_rejects_files_that_are_not_readable_pdfs(
    documents: Path, content: bytes
) -> None:
    """Nothing is written or indexed for an unusable upload."""
    with pytest.raises(InvalidDocumentError):
        document_service.upload_pdf("bad.pdf", content, documents)

    assert not documents.exists()
    assert vector_repository.document_ids() == set()


def test_removing_a_pdf_document_deletes_its_file(documents: Path) -> None:
    """Otherwise the next sync would index the deleted document again."""
    result = document_service.upload_pdf("migraine.pdf", MIGRAINE_PDF.read_bytes(), documents)

    removed = document_service.remove_document(str(result["document_id"]), documents)

    assert removed == result["chunk_count"]
    assert list(documents.iterdir()) == []
