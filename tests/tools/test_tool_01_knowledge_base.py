"""Tests for Tool 01: indexing into, searching and removing from the knowledge base."""

from __future__ import annotations

import pytest

from src.tools.tool_01_knowledge_base import service as tool_01

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")


def test_index_then_retrieve_finds_the_document() -> None:
    """A freshly indexed document is searchable immediately."""
    tool_01.index_document("Migraine", "Migraine is a headache disorder.", ["neurology"])

    results = tool_01.retrieve("migraine")

    assert results, "the indexed document should be retrievable"
    assert results[0]["title"] == "Migraine"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_retrieve_drops_matches_below_the_score_floor() -> None:
    """An unrelated query returns nothing rather than the closest weak match."""
    tool_01.index_document("Migraine", "Migraine is a headache disorder.")

    assert tool_01.retrieve("asthma") == []


def test_retrieve_orders_by_score() -> None:
    """Stronger matches come first."""
    tool_01.index_document("Aspirin", "Aspirin aspirin aspirin dosing.")
    tool_01.index_document("Mixed", "Aspirin and asthma and asthma and asthma.")

    results = tool_01.retrieve("aspirin")

    assert len(results) >= 1
    assert results[0]["title"] == "Aspirin"
    assert all(
        results[index]["score"] >= results[index + 1]["score"]
        for index in range(len(results) - 1)
    )


def test_remove_document_makes_it_unsearchable() -> None:
    """Deleting a document removes its chunks from the index."""
    result = tool_01.index_document("Migraine", "Migraine is a headache disorder.")
    assert tool_01.retrieve("migraine")

    removed = tool_01.remove_document(str(result["document_id"]))

    assert removed == result["chunk_count"]
    assert tool_01.retrieve("migraine") == []
    assert tool_01.stats()["chunk_count"] == 0


def test_remove_unknown_document_is_a_no_op() -> None:
    """Deleting something absent reports zero rather than failing."""
    tool_01.index_document("Migraine", "Migraine is a headache disorder.")

    assert tool_01.remove_document("does-not-exist") == 0
    assert tool_01.stats()["chunk_count"] == 1


def test_stats_counts_documents_and_chunks() -> None:
    """Statistics distinguish documents from the chunks they produced."""
    tool_01.index_document("Migraine", "Migraine is a headache disorder.")
    tool_01.index_document("Aspirin", "Aspirin is an analgesic.")

    assert tool_01.stats() == {"document_count": 2, "chunk_count": 2}


def test_retrieve_on_empty_index_returns_nothing() -> None:
    """An empty index yields no results and raises nothing."""
    assert tool_01.retrieve("migraine") == []


def test_index_survives_a_reload_from_disk() -> None:
    """Vectors and metadata persist, so a restart keeps the knowledge."""
    tool_01.index_document("Migraine", "Migraine is a headache disorder.")

    tool_01.load()

    results = tool_01.retrieve("migraine")
    assert results and results[0]["title"] == "Migraine"
