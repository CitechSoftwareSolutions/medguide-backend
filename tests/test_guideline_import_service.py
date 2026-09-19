"""Unit tests for bulk-importing a structured guideline document."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.dto import ImportGuidelineRequest
from src.enums import KnowledgeType
from src.models import KnowledgeEntry
from src.services import guideline_import_service


def make_payload() -> ImportGuidelineRequest:
    """Build a small guideline with one condition and shared context."""
    return ImportGuidelineRequest.model_validate(
        {
            "document_metadata": {"title": "Tremor Guide", "category": "Neurology"},
            "workflow_steps": ["1. Take history", "2. Examine"],
            "conditions_registry": [
                {
                    "condition_name": "Essential Tremor",
                    "clinical_presentation": {"key_features": ["Bilateral action tremor"]},
                    "diagnostic_approach": {"common_causes": ["Idiopathic"]},
                    "management_plan": {
                        "initial_conservative": ["Reassurance"],
                        "specific_advanced": ["Propranolol"],
                        "follow_up_monitoring": ["Review in 6 weeks"],
                    },
                }
            ],
            "global_red_flags": ["Acute onset"],
            "clinical_best_practices": ["Start low, go slow"],
            "disclaimer": "For guidance only.",
        }
    )


def make_entry(title: str) -> KnowledgeEntry:
    """Build an ORM object without requiring a database connection."""
    return KnowledgeEntry(
        id=uuid4(),
        title=title,
        summary="existing summary",
        knowledge_type=KnowledgeType.CONDITION,
        tags=[],
        created_at=datetime.now(UTC),
    )


def test_import_creates_one_entry_per_condition_plus_overview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A condition and the shared context each become a searchable entry."""
    saved_entries: list[KnowledgeEntry] = []
    indexed: list[tuple[str, str, list[str]]] = []

    monkeypatch.setattr(guideline_import_service, "find_by_title", lambda _: None)
    monkeypatch.setattr(
        guideline_import_service,
        "save",
        lambda entry: saved_entries.append(entry) or entry,
    )
    monkeypatch.setattr(
        guideline_import_service,
        "index_document",
        lambda title, text, tags: indexed.append((title, text, tags)) or {"chunk_count": 1},
    )

    results = guideline_import_service.import_guideline(make_payload())

    assert [result["title"] for result in results] == ["Essential Tremor", "Tremor Guide"]
    assert all(result["created"] for result in results)
    assert [entry.title for entry in saved_entries] == ["Essential Tremor", "Tremor Guide"]
    assert saved_entries[0].knowledge_type == KnowledgeType.CONDITION
    assert saved_entries[1].knowledge_type == KnowledgeType.PROCEDURE
    assert saved_entries[0].tags == ["neurology"]

    condition_summary = indexed[0][1]
    assert "Bilateral action tremor" in condition_summary
    assert "Propranolol" in condition_summary

    overview_summary = indexed[1][1]
    assert "Take history" in overview_summary
    assert "Acute onset" in overview_summary
    assert "For guidance only." in overview_summary


def test_import_reuses_existing_entry_by_title(monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-importing the same guideline does not duplicate a condition's entry."""
    existing = make_entry("Essential Tremor")
    monkeypatch.setattr(
        guideline_import_service,
        "find_by_title",
        lambda title: existing if title == "Essential Tremor" else None,
    )
    saved_entries: list[KnowledgeEntry] = []
    indexed_titles: list[str] = []
    monkeypatch.setattr(
        guideline_import_service,
        "save",
        lambda entry: saved_entries.append(entry) or entry,
    )
    monkeypatch.setattr(
        guideline_import_service,
        "index_document",
        lambda title, text, tags: indexed_titles.append(title) or {"chunk_count": 1},
    )

    results = guideline_import_service.import_guideline(make_payload())

    condition_result = next(r for r in results if r["title"] == "Essential Tremor")
    assert condition_result["created"] is False
    assert condition_result["id"] == existing.id
    assert condition_result["chunk_count"] == 0
    # Only the overview entry ("Tremor Guide") is new; the condition is reused.
    assert [entry.title for entry in saved_entries] == ["Tremor Guide"]
    # A reused entry must not be re-indexed, or every re-paste would double its
    # chunks in the vector store even though the knowledge-entry row is reused.
    assert indexed_titles == ["Tremor Guide"]


def test_import_skips_condition_with_no_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """A condition with none of its nested fields populated is not imported."""
    payload = ImportGuidelineRequest.model_validate(
        {
            "document_metadata": {"title": "Empty Guide"},
            "conditions_registry": [{"condition_name": "Blank Condition"}],
        }
    )
    monkeypatch.setattr(guideline_import_service, "find_by_title", lambda _: None)
    monkeypatch.setattr(
        guideline_import_service, "save", lambda entry: entry
    )
    monkeypatch.setattr(
        guideline_import_service,
        "index_document",
        lambda title, text, tags: {"chunk_count": 1},
    )

    results = guideline_import_service.import_guideline(payload)

    assert results == []


def test_request_rejects_document_with_no_content() -> None:
    """A document with neither conditions nor workflow steps is invalid."""
    with pytest.raises(ValueError):
        ImportGuidelineRequest.model_validate({"document_metadata": {"title": "Empty"}})


def test_seeder_sample_guidelines_are_valid() -> None:
    """Every pasted sample document parses and yields usable entries.

    Guards the paste-in workflow: a malformed document in the seeder would
    otherwise only surface as a failure at application startup.
    """
    from src.seeders.knowledge_seeder import SAMPLE_GUIDELINES

    assert SAMPLE_GUIDELINES

    for document in SAMPLE_GUIDELINES:
        payload = ImportGuidelineRequest.model_validate(document)
        requests = guideline_import_service.to_knowledge_requests(payload)

        assert requests, f"'{payload.document_metadata.title}' produced no entries"
        assert all(request.summary.strip() for request in requests)
        assert len({request.title for request in requests}) == len(requests)
