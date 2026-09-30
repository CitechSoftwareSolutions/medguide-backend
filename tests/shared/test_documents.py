"""Tests for the shared PDF handling, run through Tool 01's corpus."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.shared import documents
from src.shared.errors import DuplicateDocumentError, InvalidDocumentError
from src.tools.tool_01_knowledge_base import service as tool_01
from tests.sample_pdfs import DIABETES_PDF, MIGRAINE_PDF, TREMOR_PDF

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")

def test_extraction_drops_running_headers_and_page_numbers() -> None:
    """Page furniture would otherwise be embedded into every chunk."""
    text = documents.extract_pdf_text(MIGRAINE_PDF)

    assert "ICHD-3" in text
    assert "Page 1 of" not in text
    assert text.count("Clinical Teaching Note") <= 1

def test_title_comes_from_the_file_name() -> None:
    """Underscores, the .docx remnant and a copy suffix are cleaned away."""
    path = Path("Dizziness_in_Elderly_Assessment_and_Management_1.docx.pdf")

    assert documents.document_title(path) == (
        "Dizziness in Elderly Assessment and Management"
    )

def test_sync_indexes_each_pdf_once(tmp_path: Path) -> None:
    """A second sync over unchanged files embeds nothing."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    shutil.copy(TREMOR_PDF, tmp_path)

    first = tool_01.sync(tmp_path)
    chunks_after_first = tool_01.stats()["chunk_count"]
    second = tool_01.sync(tmp_path)

    assert first == {"added": 2, "removed": 0}
    assert second == {"added": 0, "removed": 0}
    assert tool_01.stats() == {
        "document_count": 2,
        "chunk_count": chunks_after_first,
    }

def test_sync_indexes_every_pdf_that_shares_a_title(tmp_path: Path) -> None:
    """Documents are told apart by content, so same-titled PDFs all go in."""
    shutil.copy(MIGRAINE_PDF, tmp_path / "Guideline.pdf")
    shutil.copy(TREMOR_PDF, tmp_path / "Guideline_1.pdf")
    shutil.copy(DIABETES_PDF, tmp_path / "Guideline 2.pdf")

    result = tool_01.sync(tmp_path)

    assert result == {"added": 3, "removed": 0}
    assert tool_01.stats()["document_count"] == 3

def test_title_keeps_a_year_suffix() -> None:
    """Only short copy suffixes are stripped from the title."""
    assert documents.document_title(Path("Diabetes_2024.pdf")) == "Diabetes 2024"

def test_sync_drops_removed_pdfs_but_keeps_api_documents(tmp_path: Path) -> None:
    """Only chunks that came from the documents folder are pruned."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    tool_01.sync(tmp_path)
    api_document = tool_01.index_document("Asthma", "Asthma is a chronic disease.")

    (tmp_path / MIGRAINE_PDF.name).unlink()
    result = tool_01.sync(tmp_path)

    assert result == {"added": 0, "removed": 1}
    assert tool_01.STORE.document_ids() == {api_document["document_id"]}

def test_indexed_pdf_is_retrievable(tmp_path: Path) -> None:
    """A question about the PDF's subject finds its chunks."""
    shutil.copy(MIGRAINE_PDF, tmp_path)
    tool_01.sync(tmp_path)

    results = tool_01.retrieve("migraine aura")

    assert results
    assert results[0]["title"] == "Migraine Diagnosis and Management"

@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """A documents folder apart from the vector store the fixture creates."""
    return tmp_path / "docs"

def test_upload_saves_and_indexes_the_pdf(folder: Path) -> None:
    """An upload is searchable immediately and lands in the documents folder."""
    result = tool_01.upload_pdf(
        "Migraine_Diagnosis_and_Management.pdf", MIGRAINE_PDF.read_bytes(), folder
    )

    assert (folder / "Migraine_Diagnosis_and_Management.pdf").exists()
    assert result["title"] == "Migraine Diagnosis and Management"
    assert result["chunk_count"] > 0
    assert tool_01.STORE.document_ids() == {result["document_id"]}
    # The startup sync must treat the uploaded file as already indexed.
    assert tool_01.sync(folder) == {"added": 0, "removed": 0}

def test_upload_rejects_the_same_pdf_twice(folder: Path) -> None:
    """Identical content under any name is a duplicate."""
    content = MIGRAINE_PDF.read_bytes()
    tool_01.upload_pdf("migraine.pdf", content, folder)

    with pytest.raises(DuplicateDocumentError):
        tool_01.upload_pdf("another-name.pdf", content, folder)

def test_upload_with_an_existing_name_keeps_both_documents(folder: Path) -> None:
    """Nothing is replaced: the second PDF is saved alongside under a suffix."""
    first = tool_01.upload_pdf("guideline.pdf", MIGRAINE_PDF.read_bytes(), folder)
    second = tool_01.upload_pdf("guideline.pdf", TREMOR_PDF.read_bytes(), folder)
    third = tool_01.upload_pdf("guideline.pdf", DIABETES_PDF.read_bytes(), folder)

    assert [first["file_name"], second["file_name"], third["file_name"]] == [
        "guideline.pdf",
        "guideline_2.pdf",
        "guideline_3.pdf",
    ]
    assert first["title"] == second["title"] == third["title"] == "guideline"
    assert tool_01.STORE.document_ids() == {
        first["document_id"],
        second["document_id"],
        third["document_id"],
    }
    # A restart finds all three already indexed and removes none of them.
    assert tool_01.sync(folder) == {
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
    folder: Path, file_name: str, saved_as: str
) -> None:
    """Directory parts are dropped and unusual characters replaced."""
    result = tool_01.upload_pdf(file_name, MIGRAINE_PDF.read_bytes(), folder)

    assert result["file_name"] == saved_as
    assert [path.name for path in folder.iterdir()] == [saved_as]

@pytest.mark.parametrize(
    "content", [b"plain text, not a pdf", b"%PDF-1.7 truncated garbage"]
)
def test_upload_rejects_files_that_are_not_readable_pdfs(
    folder: Path, content: bytes
) -> None:
    """Nothing is written or indexed for an unusable upload."""
    with pytest.raises(InvalidDocumentError):
        tool_01.upload_pdf("bad.pdf", content, folder)

    assert not folder.exists()
    assert tool_01.STORE.document_ids() == set()

def test_removing_a_pdf_document_deletes_its_file(folder: Path) -> None:
    """Otherwise the next sync would index the deleted document again."""
    result = tool_01.upload_pdf("migraine.pdf", MIGRAINE_PDF.read_bytes(), folder)

    removed = tool_01.remove_document(str(result["document_id"]), folder)

    assert removed == result["chunk_count"]
    assert list(folder.iterdir()) == []
