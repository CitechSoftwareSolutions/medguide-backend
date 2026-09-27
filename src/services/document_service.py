"""Load the PDF guidelines in the documents directory into the vector store.

Each PDF is indexed under a document identifier derived from a hash of its
bytes. That makes syncing idempotent: an unchanged file is skipped, an edited
file gets a new identifier (its old chunks are removed), and a deleted file's
chunks are dropped. Documents indexed through the API carry ordinary random
identifiers without the ``pdf-`` prefix, so a sync never touches them.

Uploaded PDFs are written into the same directory, so the directory stays the
source of truth: an upload survives a restart and a rebuild, and deleting a PDF
document removes its file so it does not come back on the next sync. Documents
are told apart by content, never by name or title: two different PDFs that
share a title are both kept and both indexed.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import threading
from collections import Counter
from pathlib import Path
from typing import BinaryIO

from src.config import get_settings
from src.exceptions import DuplicateDocumentError, InvalidDocumentError
from src.repositories import vector_repository
from src.services import rag_service

logger = logging.getLogger(__name__)

PDF_DOCUMENT_PREFIX = "pdf-"

# Serializes everything that changes the documents directory or its chunks, so
# an upload cannot interleave with a sync and index the same file twice.
_lock = threading.Lock()

_PAGE_NUMBER = re.compile(r"\bPage\s+\d+\s+of\s+\d+\b", re.IGNORECASE)
# Bullet glyphs as they come out of the extractor; U+FFFD is a bullet whose
# font had no Unicode mapping.
_BULLET = re.compile(r"[●•▪◦‣⁃�]")
_WHITESPACE = re.compile(r"\s+")
_UNSAFE_FILE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")


def _normalize_line(line: str) -> str:
    return _WHITESPACE.sub(" ", _PAGE_NUMBER.sub("", line)).strip()


def extract_pdf_text(source: Path | BinaryIO) -> str:
    """Return a PDF's text with page furniture removed and whitespace collapsed.

    A line that repeats on at least half the pages of a multi-page document is
    a running header or footer. Left in, it would be embedded into every chunk
    and pull all of them towards the same point in vector space.
    """
    from pypdf import PdfReader

    pages = [
        [_normalize_line(line) for line in (page.extract_text() or "").splitlines()]
        for page in PdfReader(source).pages
    ]
    pages = [[line for line in lines if line] for lines in pages]

    repeated: set[str] = set()
    if len(pages) > 1:
        counts = Counter(line for lines in pages for line in set(lines))
        repeated = {line for line, count in counts.items() if count * 2 >= len(pages)}

    text = " ".join(line for lines in pages for line in lines if line not in repeated)
    # Bullets become "• " so the chunker can split on them; list items often
    # carry no terminal punctuation.
    text = _BULLET.sub(" • ", text)
    return _WHITESPACE.sub(" ", text).strip()


def document_title(path: Path) -> str:
    """Derive a readable title from a file name.

    ``Dizziness_in_Elderly_Assessment_and_Management_1.docx.pdf`` becomes
    ``Dizziness in Elderly Assessment and Management``. Only a one- or two-digit
    copy suffix is dropped, so a year such as ``Diabetes_2024`` survives.
    """
    stem = path.name
    for suffix in (".pdf", ".docx"):
        stem = stem.removesuffix(suffix)
    stem = re.sub(r"[_\s]+\d{1,2}$", "", stem)
    return _WHITESPACE.sub(" ", stem.replace("_", " ")).strip()


def pdf_document_id(content: bytes) -> str:
    """Return the content-derived identifier for one PDF's bytes."""
    return f"{PDF_DOCUMENT_PREFIX}{hashlib.sha256(content).hexdigest()[:32]}"


def _safe_file_name(file_name: str) -> str:
    """Reduce a client-supplied name to a plain ``.pdf`` name in one directory.

    ``Path(...).name`` drops any directory part, so ``../../main.py`` cannot
    escape the documents directory.
    """
    stem = Path(file_name.replace("\\", "/")).name.removesuffix(".pdf").removesuffix(".PDF")
    stem = _UNSAFE_FILE_NAME_CHARACTERS.sub("_", stem).strip(" .")
    return f"{stem or 'document'}.pdf"


def _unused_path(directory: Path, file_name: str) -> Path:
    """Return ``file_name`` in ``directory``, suffixed ``_2``, ``_3``... if taken.

    The numeric suffix is the one :func:`document_title` strips, so every copy
    keeps the title of the original name.
    """
    target = directory / file_name
    stem = target.stem
    counter = 2
    while target.exists():
        target = directory / f"{stem}_{counter}.pdf"
        counter += 1
    return target


def _files_by_document_id(directory: Path) -> dict[str, Path]:
    paths = sorted(directory.glob("*.pdf")) if directory.is_dir() else []
    return {pdf_document_id(path.read_bytes()): path for path in paths}


def _indexed_pdf_ids() -> set[str]:
    return {
        document_id
        for document_id in vector_repository.document_ids()
        if document_id.startswith(PDF_DOCUMENT_PREFIX)
    }


def _sync_unlocked(directory: Path) -> dict[str, int]:
    wanted = _files_by_document_id(directory)
    if not wanted:
        logger.warning("No PDF documents found in %s", directory)
    indexed = _indexed_pdf_ids()

    removed = 0
    for document_id in indexed - wanted.keys():
        removed += 1
        rag_service.remove_document(document_id)

    added = 0
    for document_id, path in wanted.items():
        if document_id in indexed:
            continue
        text = extract_pdf_text(path)
        if not text:
            logger.warning("No extractable text in %s; is it a scanned PDF?", path.name)
            continue
        result = rag_service.index_document(
            document_title(path), text, ["pdf"], document_id=document_id
        )
        added += 1
        logger.info("Indexed %s as %s chunks", path.name, result["chunk_count"])

    logger.info(
        "Document sync: %d added, %d removed, %d unchanged",
        added,
        removed,
        len(wanted.keys() & indexed),
    )
    return {"added": added, "removed": removed}


def sync_documents_directory(directory: Path | None = None) -> dict[str, int]:
    """Bring the index in line with the PDFs in the documents directory."""
    with _lock:
        return _sync_unlocked(directory or get_settings().documents_dir)


def rebuild_from_documents_directory(directory: Path | None = None) -> dict[str, int]:
    """Drop every vector and index the documents directory from scratch."""
    with _lock:
        vector_repository.reset()
        _sync_unlocked(directory or get_settings().documents_dir)
    return vector_repository.stats()


def upload_pdf(
    file_name: str, content: bytes, directory: Path | None = None
) -> dict[str, object]:
    """Save an uploaded PDF into the documents directory and index it.

    An existing document is never replaced or removed. A different PDF that
    arrives under a name already in use is saved under a suffixed name and
    indexed alongside the first; only byte-identical content is rejected, since
    indexing it again would add nothing but duplicate chunks.
    """
    directory = directory or get_settings().documents_dir
    if not content.startswith(b"%PDF-"):
        raise InvalidDocumentError("The uploaded file is not a PDF.")

    from pypdf.errors import PyPdfError

    try:
        text = extract_pdf_text(io.BytesIO(content))
    except PyPdfError as exc:
        raise InvalidDocumentError("The PDF could not be read.") from exc
    if not text:
        raise InvalidDocumentError(
            "The PDF has no extractable text. Scanned PDFs need OCR first."
        )

    safe_name = _safe_file_name(file_name)
    document_id = pdf_document_id(content)
    title = document_title(Path(safe_name))

    with _lock:
        existing = _files_by_document_id(directory)
        if document_id in existing or document_id in _indexed_pdf_ids():
            name = existing[document_id].name if document_id in existing else safe_name
            raise DuplicateDocumentError(
                f"This PDF is already in the knowledge base as '{name}'."
            )

        directory.mkdir(parents=True, exist_ok=True)
        target = _unused_path(directory, safe_name)
        temporary = target.with_suffix(".pdf.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)

        result = rag_service.index_document(title, text, ["pdf"], document_id=document_id)

    logger.info("Uploaded %s as %s chunks", target.name, result["chunk_count"])
    return {
        "document_id": document_id,
        "chunk_count": int(result["chunk_count"]),
        "file_name": target.name,
        "title": title,
    }


def remove_document(document_id: str, directory: Path | None = None) -> int:
    """Remove a document's chunks and, for a PDF, its file. Returns chunks removed."""
    directory = directory or get_settings().documents_dir
    with _lock:
        if document_id.startswith(PDF_DOCUMENT_PREFIX):
            path = _files_by_document_id(directory).get(document_id)
            if path is not None:
                path.unlink()
                logger.info("Deleted %s", path.name)
        return rag_service.remove_document(document_id)
