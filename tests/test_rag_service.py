"""Tests for chunking, indexing and retrieval."""

from __future__ import annotations

import pytest

from src.config.settings import get_settings
from src.services import rag_service

pytestmark = pytest.mark.usefixtures("isolated_vector_store", "fake_embeddings")


def test_chunk_prepends_title_to_every_chunk() -> None:
    """Chunks must carry their subject so they stay retrievable in isolation."""
    text = " ".join(f"Sentence number {index} about aura." for index in range(40))

    chunks = rag_service.chunk_text("Migraine", text)

    assert len(chunks) > 1, "a long document should split into several chunks"
    assert all(chunk.startswith("Migraine\n\n") for chunk in chunks)


def test_chunk_respects_the_configured_size() -> None:
    """Chunks stay near the configured size rather than growing unbounded."""
    settings = get_settings()
    text = " ".join(f"Clinical detail {index}." for index in range(80))

    chunks = rag_service.chunk_text("Aspirin", text)

    body_lengths = [len(chunk) - len("Aspirin\n\n") for chunk in chunks]
    # A chunk may exceed the target by the final sentence that tripped the
    # boundary, plus the overlap carried forward from the previous chunk.
    allowance = settings.chunk_size + settings.chunk_overlap + 60
    assert max(body_lengths) <= allowance


def test_chunk_returns_nothing_for_blank_text() -> None:
    """Whitespace-only input produces no chunks rather than one empty chunk."""
    assert rag_service.chunk_text("Empty", "   \n  ") == []


def test_consecutive_chunks_overlap_on_whole_words() -> None:
    """Overlap keeps boundary-spanning facts findable without word fragments."""
    text = " ".join(f"Clinical detail {index} recorded." for index in range(60))

    chunks = rag_service.chunk_text("Aspirin", text)
    bodies = [chunk.removeprefix("Aspirin\n\n") for chunk in chunks]

    assert len(bodies) > 1
    # The second chunk opens with text carried over from the first.
    carried = bodies[1].split(".")[0]
    assert carried and carried in bodies[0]
    # And it opens on a word boundary, not halfway through one.
    assert not bodies[1].startswith(" ")
    first_word = bodies[1].split(" ", 1)[0].rstrip(".")
    assert first_word in {"Clinical", "detail", "recorded"} or first_word.isdigit()


@pytest.mark.parametrize(
    ("text", "overlap", "expected"),
    [
        ("alpha beta gamma", 8, "gamma"),
        ("alpha beta gamma", 0, ""),
        ("short", 50, "short"),
    ],
)
def test_overlap_tail_trims_to_a_word_boundary(
    text: str, overlap: int, expected: str
) -> None:
    """The carried tail never begins mid-word."""
    assert rag_service._overlap_tail(text, overlap) == expected


def test_index_then_retrieve_finds_the_document() -> None:
    """A freshly indexed document is searchable immediately."""
    rag_service.index_document("Migraine", "Migraine is a headache disorder.", ["neurology"])

    results = rag_service.retrieve("migraine")

    assert results, "the indexed document should be retrievable"
    assert results[0]["title"] == "Migraine"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-5)


def test_retrieve_drops_matches_below_the_score_floor() -> None:
    """An unrelated query returns nothing rather than the closest weak match."""
    rag_service.index_document("Migraine", "Migraine is a headache disorder.")

    assert rag_service.retrieve("asthma") == []


def test_retrieve_orders_by_score() -> None:
    """Stronger matches come first."""
    rag_service.index_document("Aspirin", "Aspirin aspirin aspirin dosing.")
    rag_service.index_document("Mixed", "Aspirin and asthma and asthma and asthma.")

    results = rag_service.retrieve("aspirin")

    assert len(results) >= 1
    assert results[0]["title"] == "Aspirin"
    assert all(
        results[index]["score"] >= results[index + 1]["score"]
        for index in range(len(results) - 1)
    )


def test_remove_document_makes_it_unsearchable() -> None:
    """Deleting a document removes its chunks from the index."""
    result = rag_service.index_document("Migraine", "Migraine is a headache disorder.")
    assert rag_service.retrieve("migraine")

    removed = rag_service.remove_document(str(result["document_id"]))

    assert removed == result["chunk_count"]
    assert rag_service.retrieve("migraine") == []
    assert rag_service.stats()["chunk_count"] == 0


def test_remove_unknown_document_is_a_no_op() -> None:
    """Deleting something absent reports zero rather than failing."""
    rag_service.index_document("Migraine", "Migraine is a headache disorder.")

    assert rag_service.remove_document("does-not-exist") == 0
    assert rag_service.stats()["chunk_count"] == 1


def test_stats_counts_documents_and_chunks() -> None:
    """Statistics distinguish documents from the chunks they produced."""
    rag_service.index_document("Migraine", "Migraine is a headache disorder.")
    rag_service.index_document("Aspirin", "Aspirin is an analgesic.")

    assert rag_service.stats() == {"document_count": 2, "chunk_count": 2}


def test_retrieve_on_empty_index_returns_nothing() -> None:
    """An empty index yields no results and raises nothing."""
    assert rag_service.retrieve("migraine") == []


def test_index_survives_a_reload_from_disk() -> None:
    """Vectors and metadata persist, so a restart keeps the knowledge."""
    from src.repositories import vector_repository

    rag_service.index_document("Migraine", "Migraine is a headache disorder.")

    vector_repository.load_or_create()

    results = rag_service.retrieve("migraine")
    assert results and results[0]["title"] == "Migraine"
