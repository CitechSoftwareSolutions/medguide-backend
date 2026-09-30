"""Tests for chunking, shared by both tools."""

from __future__ import annotations

import pytest

from src.shared import indexing
from src.shared.config import get_settings



def test_chunk_prepends_title_to_every_chunk() -> None:
    """Chunks must carry their subject so they stay retrievable in isolation."""
    text = " ".join(f"Sentence number {index} about aura." for index in range(40))

    chunks = indexing.chunk_text("Migraine", text)

    assert len(chunks) > 1, "a long document should split into several chunks"
    assert all(chunk.startswith("Migraine\n\n") for chunk in chunks)


def test_chunk_respects_the_configured_size() -> None:
    """Chunks stay near the configured size rather than growing unbounded."""
    settings = get_settings()
    text = " ".join(f"Clinical detail {index}." for index in range(80))

    chunks = indexing.chunk_text("Aspirin", text)

    body_lengths = [len(chunk) - len("Aspirin\n\n") for chunk in chunks]
    # A chunk may exceed the target by the final sentence that tripped the
    # boundary, plus the overlap carried forward from the previous chunk.
    allowance = settings.chunk_size + settings.chunk_overlap + 60
    assert max(body_lengths) <= allowance


def test_chunk_returns_nothing_for_blank_text() -> None:
    """Whitespace-only input produces no chunks rather than one empty chunk."""
    assert indexing.chunk_text("Empty", "   \n  ") == []


def test_consecutive_chunks_overlap_on_whole_words() -> None:
    """Overlap keeps boundary-spanning facts findable without word fragments."""
    text = " ".join(f"Clinical detail {index} recorded." for index in range(60))

    chunks = indexing.chunk_text("Aspirin", text)
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
    assert indexing._overlap_tail(text, overlap) == expected
