"""Chunking, indexing and retrieval over the medical knowledge base."""

from __future__ import annotations

import logging
import re

from src.agent.llm import encode_documents, encode_query
from src.agent.state import RetrievedChunk
from src.config import get_settings
from src.repositories import list_all, vector_repository
from src.repositories.vector_repository import ChunkRecord
from src.utils import new_identifier

logger = logging.getLogger(__name__)

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, tolerating text with no terminal punctuation."""
    return [part.strip() for part in _SENTENCE_BOUNDARY.split(text.strip()) if part.strip()]


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


def index_document(title: str, text: str, tags: list[str] | None = None) -> dict[str, object]:
    """Index one document and return its identifier and chunk count."""
    document_id = str(new_identifier())
    pieces = chunk_text(title, text)
    if not pieces:
        return {"document_id": document_id, "chunk_count": 0}

    records: list[ChunkRecord] = [
        {
            "chunk_id": str(new_identifier()),
            "document_id": document_id,
            "title": title,
            "text": piece,
            "tags": list(tags or []),
        }
        for piece in pieces
    ]
    vectors = encode_documents([record["text"] for record in records])
    vector_repository.add_chunks(records, vectors)
    logger.info("Indexed document %s as %d chunks", document_id, len(records))
    return {"document_id": document_id, "chunk_count": len(records)}


def remove_document(document_id: str) -> int:
    """Remove one document's chunks from the index."""
    return vector_repository.remove_document(document_id)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the chunks most similar to a query, above the score floor.

    Returns an empty list when nothing clears the floor. Callers decide what an
    empty result means; an empty *index* is a different condition and is
    reported by :func:`stats`.
    """
    settings = get_settings()
    vector = encode_query(query)
    matches = vector_repository.search(vector, top_k or settings.retrieval_top_k)

    return [
        RetrievedChunk(
            chunk_id=record["chunk_id"],
            document_id=record["document_id"],
            title=record["title"],
            text=record["text"],
            score=score,
        )
        for record, score in matches
        if score >= settings.retrieval_min_score
    ]


def stats() -> dict[str, int]:
    """Report how much knowledge is searchable."""
    return vector_repository.stats()


def reindex_from_knowledge_entries() -> dict[str, int]:
    """Rebuild the index from the persisted knowledge entries.

    Reuses the existing ``list_all`` repository function so the seeded entries
    become searchable without duplicating any query logic.
    """
    vector_repository.reset()
    documents = 0
    chunks = 0
    for entry in list_all():
        result = index_document(entry.title, entry.summary, list(entry.tags or []))
        documents += 1
        chunks += int(result["chunk_count"])
    logger.info("Reindexed %d documents into %d chunks", documents, chunks)
    return {"document_count": documents, "chunk_count": chunks}
