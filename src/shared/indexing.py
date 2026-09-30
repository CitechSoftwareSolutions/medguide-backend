"""Chunk, embed, store and search text in any :class:`VectorStore`.

Both tools index and search through these functions, so the knowledge base
(Tool 01) and NICE guidance (Tool 02) are chunked and embedded identically.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict
from uuid import uuid4

from src.shared.config import get_settings
from src.shared.embeddings import encode_documents, encode_query
from src.shared.vector_store import ChunkRecord, VectorStore

logger = logging.getLogger(__name__)

# Also split before a bullet: list items extracted from PDFs often carry no
# terminal punctuation and would otherwise merge into one oversized "sentence".
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\s+(?=• )")


class RetrievedChunk(TypedDict):
    """A chunk returned by vector search, with its cosine score."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, tolerating text with no terminal punctuation."""
    parts = (part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()))
    return [part for part in parts if part and part != "•"]


def _overlap_tail(text: str, overlap: int) -> str:
    """Return the trailing overlap, trimmed to a whole-word boundary."""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    # A tail that begins mid-word embeds a meaningless fragment, so drop the
    # partial first word unless doing so would leave nothing.
    if len(tail) < len(text) and " " in tail:
        tail = tail[tail.index(" ") + 1 :]
    return tail.strip()


def chunk_text(title: str, text: str) -> list[str]:
    """Split a document into overlapping, sentence-aligned chunks.

    Every chunk is prefixed with the document title. Short clinical summaries
    routinely omit their own subject ("Usually presents with aura...") and lose
    all retrievability once detached from the entry they belong to; carrying the
    title into the embedded text is the single largest accuracy win here.
    """
    settings = get_settings()
    size = settings.chunk_size
    overlap = settings.chunk_overlap
    prefix = f"{title}\n\n"

    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and len(candidate) > size:
            chunks.append(prefix + current)
            # Carry the tail of the finished chunk forward so a fact spanning a
            # boundary stays retrievable from both sides.
            carry = _overlap_tail(current, overlap)
            current = f"{carry} {sentence}".strip() if carry else sentence
        else:
            current = candidate

    if current:
        chunks.append(prefix + current)
    return chunks


def index_into(
    store: VectorStore,
    title: str,
    text: str,
    tags: list[str] | None = None,
    *,
    document_id: str | None = None,
) -> dict[str, object]:
    """Index one document into ``store`` and return its identifier and chunk count.

    ``document_id`` is generated unless the caller needs a stable one.
    """
    document_id = document_id or str(uuid4())
    pieces = chunk_text(title, text)
    if not pieces:
        return {"document_id": document_id, "chunk_count": 0}

    records: list[ChunkRecord] = [
        {
            "chunk_id": str(uuid4()),
            "document_id": document_id,
            "title": title,
            "text": piece,
            "tags": list(tags or []),
        }
        for piece in pieces
    ]
    vectors = encode_documents([record["text"] for record in records])
    store.add_chunks(records, vectors)
    logger.info(
        "Indexed document %s into the %s store as %d chunks",
        document_id,
        store.name,
        len(records),
    )
    return {"document_id": document_id, "chunk_count": len(records)}


def search_store(
    store: VectorStore, query: str, top_k: int, min_score: float
) -> list[RetrievedChunk]:
    """Return the chunks in ``store`` most similar to a query, above ``min_score``.

    Returns an empty list when nothing clears the floor. Callers decide what an
    empty result means; an empty *store* is a different condition, reported by
    ``store.stats()``.
    """
    vector = encode_query(query)
    return [
        RetrievedChunk(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            title=record["title"],
            text=record["text"],
            score=score,
        )
        for record, score in store.search(vector, top_k)
        if score >= min_score
    ]
