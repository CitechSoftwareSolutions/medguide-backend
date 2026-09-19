"""FAISS-backed persistence for knowledge chunks.

The index is ``IndexIDMap2`` wrapping ``IndexFlatIP``. Two consequences matter:

* ``IndexFlatIP`` over L2-normalized vectors is *exact* cosine similarity. There
  is no approximate-nearest-neighbour step, so identical input always yields
  identical results. Revisit ``IndexHNSWFlat`` only past roughly a million
  vectors, where the exhaustive scan stops being cheap.
* ``IndexIDMap2`` supplies ``add_with_ids`` and ``remove_ids``, which is what
  lets the owner add and delete documents at runtime without a full rebuild.

FAISS is not safe for concurrent reads and writes, so one module-level lock
guards every entry point. Searches here are sub-millisecond, so a single lock is
correct and cheap; copy-on-write index swapping would only be worth it if
profiling ever showed contention.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_index: Any | None = None
_metadata: dict[int, "ChunkRecord"] = {}
_next_faiss_id: int = 0


class ChunkRecord(TypedDict):
    """Everything stored alongside a vector."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    tags: list[str]


def _index_path() -> Path:
    return get_settings().vector_store_dir / "index.faiss"


def _metadata_path() -> Path:
    return get_settings().vector_store_dir / "chunks.json"


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


def _load_unlocked() -> None:
    """Populate the module-level index and metadata from disk."""
    global _index, _metadata, _next_faiss_id

    import faiss

    index_path = _index_path()
    metadata_path = _metadata_path()

    if index_path.exists() and metadata_path.exists():
        _index = faiss.read_index(str(index_path))
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        _metadata = {int(key): value for key, value in raw.items()}
        _next_faiss_id = max(_metadata, default=-1) + 1
        logger.info("Loaded vector store with %d chunks", len(_metadata))
        return

    _index = _new_index()
    _metadata = {}
    _next_faiss_id = 0
    logger.info("Created an empty vector store")


def _persist_unlocked() -> None:
    """Flush the index and its metadata sidecar to disk."""
    import faiss

    _write_atomically(_index_path(), lambda path: faiss.write_index(_index, str(path)))
    _write_atomically(
        _metadata_path(),
        lambda path: path.write_text(
            json.dumps({str(key): value for key, value in _metadata.items()}),
            encoding="utf-8",
        ),
    )


def load_or_create() -> None:
    """Load the store from disk, creating an empty one when absent."""
    with _lock:
        _load_unlocked()


def _ensure_loaded_unlocked() -> None:
    if _index is None:
        _load_unlocked()


def add_chunks(records: list[ChunkRecord], vectors: np.ndarray) -> int:
    """Add chunk vectors and their metadata, then persist. Returns the count added."""
    global _next_faiss_id

    if not records:
        return 0
    if len(records) != len(vectors):
        raise ValueError("Each chunk record needs exactly one vector.")

    with _lock:
        _ensure_loaded_unlocked()
        faiss_ids = np.arange(
            _next_faiss_id, _next_faiss_id + len(records), dtype=np.int64
        )
        _index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), faiss_ids)
        for faiss_id, record in zip(faiss_ids, records, strict=True):
            _metadata[int(faiss_id)] = record
        _next_faiss_id += len(records)
        _persist_unlocked()

    return len(records)


def remove_document(document_id: str) -> int:
    """Remove every chunk belonging to one document. Returns the count removed."""
    with _lock:
        _ensure_loaded_unlocked()
        doomed = [
            faiss_id
            for faiss_id, record in _metadata.items()
            if record["document_id"] == document_id
        ]
        if not doomed:
            return 0
        # The FAISS Python wrapper accepts an array of ids here and builds the
        # selector itself; constructing an IDSelector by hand is error-prone.
        _index.remove_ids(np.array(doomed, dtype=np.int64))
        for faiss_id in doomed:
            del _metadata[faiss_id]
        _persist_unlocked()

    return len(doomed)


def search(vector: np.ndarray, top_k: int) -> list[tuple[ChunkRecord, float]]:
    """Return the closest chunks with their cosine scores, best first."""
    with _lock:
        _ensure_loaded_unlocked()
        if _index.ntotal == 0:
            return []
        query = np.ascontiguousarray(
            vector.reshape(1, -1), dtype=np.float32
        )
        scores, ids = _index.search(query, min(top_k, _index.ntotal))
        results: list[tuple[ChunkRecord, float]] = []
        for faiss_id, score in zip(ids[0], scores[0], strict=True):
            # FAISS returns -1 to pad results when fewer neighbours exist.
            if faiss_id == -1:
                continue
            record = _metadata.get(int(faiss_id))
            if record is not None:
                results.append((record, float(score)))
        return results


def stats() -> dict[str, int]:
    """Report how much knowledge is currently searchable."""
    with _lock:
        _ensure_loaded_unlocked()
        return {
            "chunk_count": int(_index.ntotal),
            "document_count": len(
                {record["document_id"] for record in _metadata.values()}
            ),
        }


def reset() -> None:
    """Drop every vector. Used by reindexing and by tests."""
    global _index, _metadata, _next_faiss_id
    with _lock:
        _index = _new_index()
        _metadata = {}
        _next_faiss_id = 0
        _persist_unlocked()
