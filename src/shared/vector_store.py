"""FAISS-backed persistence for knowledge chunks.

The index is ``IndexIDMap2`` wrapping ``IndexFlatIP``. Two consequences matter:

* ``IndexFlatIP`` over L2-normalized vectors is *exact* cosine similarity. There
  is no approximate-nearest-neighbour step, so identical input always yields
  identical results. Revisit ``IndexHNSWFlat`` only past roughly a million
  vectors, where the exhaustive scan stops being cheap.
* ``IndexIDMap2`` supplies ``add_with_ids`` and ``remove_ids``, which is what
  lets the owner add and delete documents at runtime without a full rebuild.

Each tool owns one :class:`VectorStore` with its own directory, index and lock
(see ``src/tools/*/service.py``). Keeping them apart is what guarantees a NICE
passage (Tool 02) can never become answer material (Tool 01).

FAISS is not safe for concurrent reads and writes, so each store's lock guards
every entry point. Searches here are sub-millisecond, so a single lock per store
is correct and cheap; copy-on-write index swapping would only be worth it if
profiling ever showed contention.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.shared.config import get_settings

logger = logging.getLogger(__name__)


class ChunkRecord(TypedDict):
    """Everything stored alongside a vector."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    tags: list[str]


def _new_index() -> Any:
    """Create an empty exact-cosine index."""
    import faiss

    settings = get_settings()
    return faiss.IndexIDMap2(faiss.IndexFlatIP(settings.embedding_dimensions))


def _write_atomically(path: Path, write: Any) -> None:
    """Write through a temporary file so a crash cannot leave a torn store."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary)
    os.replace(temporary, path)


class VectorStore:
    """One persisted FAISS index and its metadata sidecar.

    ``directory`` is a callable rather than a path so the location is read from
    settings each time the store loads; tests repoint it through environment
    variables and a settings cache clear.
    """

    def __init__(self, name: str, directory: Callable[[], Path]) -> None:
        self.name = name
        self._directory = directory
        self._lock = threading.Lock()
        self._index: Any | None = None
        self._metadata: dict[int, ChunkRecord] = {}
        self._next_faiss_id = 0

    def _index_path(self) -> Path:
        return self._directory() / "index.faiss"

    def _metadata_path(self) -> Path:
        return self._directory() / "chunks.json"

    def _load_unlocked(self) -> None:
        """Populate the index and metadata from disk."""
        import faiss

        index_path = self._index_path()
        metadata_path = self._metadata_path()

        if index_path.exists() and metadata_path.exists():
            self._index = faiss.read_index(str(index_path))
            raw = json.loads(metadata_path.read_text(encoding="utf-8"))
            self._metadata = {int(key): value for key, value in raw.items()}
            self._next_faiss_id = max(self._metadata, default=-1) + 1
            logger.info("Loaded %s store with %d chunks", self.name, len(self._metadata))
            return

        self._index = _new_index()
        self._metadata = {}
        self._next_faiss_id = 0
        logger.info("Created an empty %s store", self.name)

    def _persist_unlocked(self) -> None:
        """Flush the index and its metadata sidecar to disk."""
        import faiss

        _write_atomically(
            self._index_path(), lambda path: faiss.write_index(self._index, str(path))
        )
        _write_atomically(
            self._metadata_path(),
            lambda path: path.write_text(
                json.dumps({str(key): value for key, value in self._metadata.items()}),
                encoding="utf-8",
            ),
        )

    def _ensure_loaded_unlocked(self) -> None:
        if self._index is None:
            self._load_unlocked()

    def load_or_create(self) -> None:
        """Load the store from disk, creating an empty one when absent."""
        with self._lock:
            self._load_unlocked()

    def add_chunks(self, records: list[ChunkRecord], vectors: np.ndarray) -> int:
        """Add chunk vectors and their metadata, then persist. Returns the count added."""
        if not records:
            return 0
        if len(records) != len(vectors):
            raise ValueError("Each chunk record needs exactly one vector.")

        with self._lock:
            self._ensure_loaded_unlocked()
            faiss_ids = np.arange(
                self._next_faiss_id, self._next_faiss_id + len(records), dtype=np.int64
            )
            self._index.add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32), faiss_ids
            )
            for faiss_id, record in zip(faiss_ids, records, strict=True):
                self._metadata[int(faiss_id)] = record
            self._next_faiss_id += len(records)
            self._persist_unlocked()

        return len(records)

    def remove_document(self, document_id: str) -> int:
        """Remove every chunk belonging to one document. Returns the count removed."""
        with self._lock:
            self._ensure_loaded_unlocked()
            doomed = [
                faiss_id
                for faiss_id, record in self._metadata.items()
                if record["document_id"] == document_id
            ]
            if not doomed:
                return 0
            # The FAISS Python wrapper accepts an array of ids here and builds
            # the selector itself; constructing an IDSelector by hand is
            # error-prone.
            self._index.remove_ids(np.array(doomed, dtype=np.int64))
            for faiss_id in doomed:
                del self._metadata[faiss_id]
            self._persist_unlocked()

        return len(doomed)

    def search(self, vector: np.ndarray, top_k: int) -> list[tuple[ChunkRecord, float]]:
        """Return the closest chunks with their cosine scores, best first."""
        with self._lock:
            self._ensure_loaded_unlocked()
            if self._index.ntotal == 0:
                return []
            query = np.ascontiguousarray(vector.reshape(1, -1), dtype=np.float32)
            scores, ids = self._index.search(query, min(top_k, self._index.ntotal))
            results: list[tuple[ChunkRecord, float]] = []
            for faiss_id, score in zip(ids[0], scores[0], strict=True):
                # FAISS returns -1 to pad results when fewer neighbours exist.
                if faiss_id == -1:
                    continue
                record = self._metadata.get(int(faiss_id))
                if record is not None:
                    results.append((record, float(score)))
            return results

    def document_ids(self) -> set[str]:
        """Return the identifier of every indexed document."""
        with self._lock:
            self._ensure_loaded_unlocked()
            return {record["document_id"] for record in self._metadata.values()}

    def list_documents(self) -> list[dict[str, Any]]:
        """Summarize each indexed document: identifier, title, tags and chunk count."""
        with self._lock:
            self._ensure_loaded_unlocked()
            documents: dict[str, dict[str, Any]] = {}
            for record in self._metadata.values():
                summary = documents.setdefault(
                    record["document_id"],
                    {
                        "document_id": record["document_id"],
                        "title": record["title"],
                        "tags": list(record["tags"]),
                        "chunk_count": 0,
                    },
                )
                summary["chunk_count"] += 1
        return sorted(
            documents.values(), key=lambda item: (item["title"].lower(), item["document_id"])
        )

    def stats(self) -> dict[str, int]:
        """Report how much is currently searchable."""
        with self._lock:
            self._ensure_loaded_unlocked()
            return {
                "chunk_count": int(self._index.ntotal),
                "document_count": len(
                    {record["document_id"] for record in self._metadata.values()}
                ),
            }

    def reset(self) -> None:
        """Drop every vector. Used by reindexing and by tests."""
        with self._lock:
            self._index = _new_index()
            self._metadata = {}
            self._next_faiss_id = 0
            self._persist_unlocked()
