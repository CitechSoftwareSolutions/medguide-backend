"""Tool 02's vector store, and indexing, search and sync over it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared import documents
from src.shared.config import get_settings
from src.shared.documents import Corpus
from src.shared.indexing import RetrievedChunk, index_into, search_store
from src.shared.vector_store import VectorStore

STORE = VectorStore("NICE guidance", lambda: get_settings().nice_vector_store_dir)
CORPUS = Corpus(
    name="NICE guidance",
    store=STORE,
    folder=lambda: get_settings().nice_documents_dir,
    # NICE file names carry download ids; the PDF metadata has the real title.
    metadata_titles=True,
)


def retrieve(query: str, top_k: int | None = None) -> list[RetrievedChunk]:
    """Return the NICE chunks most similar to a query, above the score floor.

    NICE guidelines repeat whole paragraphs across sections, so identical chunks
    are common. Twice as many candidates are fetched and repeats dropped, so the
    check sees ``top_k`` distinct passages rather than the same one twice.
    """
    settings = get_settings()
    top_k = top_k or settings.nice_top_k
    candidates = search_store(STORE, query, top_k * 2, settings.nice_min_score)
    seen: set[str] = set()
    distinct: list[RetrievedChunk] = []
    for chunk in candidates:
        if chunk["text"] not in seen:
            seen.add(chunk["text"])
            distinct.append(chunk)
    return distinct[:top_k]


def index_document(
    title: str,
    text: str,
    tags: list[str] | None = None,
    *,
    document_id: str | None = None,
) -> dict[str, object]:
    """Index text into the NICE store."""
    return index_into(STORE, title, text, tags, document_id=document_id)


def list_documents() -> list[dict[str, Any]]:
    return STORE.list_documents()


def stats() -> dict[str, int]:
    return STORE.stats()


def load() -> None:
    """Load the store from disk, creating an empty one when absent."""
    STORE.load_or_create()


def sync(directory: Path | None = None) -> dict[str, int]:
    """Index new or changed PDFs in ``data/`` and drop removed ones."""
    return documents.sync(CORPUS, directory)


def rebuild(directory: Path | None = None) -> dict[str, int]:
    return documents.rebuild(CORPUS, directory)


def upload_pdf(file_name: str, content: bytes, directory: Path | None = None) -> dict[str, object]:
    return documents.upload_pdf(CORPUS, file_name, content, directory)


def remove_document(document_id: str, directory: Path | None = None) -> int:
    return documents.remove_document(CORPUS, document_id, directory)
