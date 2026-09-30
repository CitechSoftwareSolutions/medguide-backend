"""Keep a tool's vector store in step with the PDFs in its ``data`` folder.

Each tool describes its PDFs as a :class:`Corpus`: a name, a folder and the
vector store they are indexed into. The functions here take a corpus and work
the same way for Tool 01 and Tool 02.

Each PDF is indexed under a document identifier derived from a hash of its
bytes. That makes syncing idempotent: an unchanged file is skipped, an edited
file gets a new identifier (its old chunks are removed), and a deleted file's
chunks are dropped. Documents indexed through the API carry ordinary random
identifiers without the ``pdf-`` prefix, so a sync never touches them.

Uploaded PDFs are written into the same folder, so the folder stays the source
of truth: an upload survives a restart and a rebuild, and deleting a PDF
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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from src.shared.errors import DuplicateDocumentError, InvalidDocumentError
from src.shared.indexing import index_into
from src.shared.vector_store import VectorStore

logger = logging.getLogger(__name__)

PDF_DOCUMENT_PREFIX = "pdf-"

# Serializes everything that changes a documents directory or its chunks, so
# an upload cannot interleave with a sync and index the same file twice.
_lock = threading.Lock()

# The page total is optional: NICE footers often wrap it onto the next line.
_PAGE_NUMBER = re.compile(r"\bPage\s+\d+\s+of\b(?:\s+\d+\b)?", re.IGNORECASE)
# Table-of-contents leaders ("Recommendations ........ 6"). They carry no
# content and would otherwise fill whole chunks with dots.
_DOT_LEADER = re.compile(r"\s*\.{4,}\s*\d*")
# NICE's copyright footer wraps at different points on different pages, so the
# repeated-line filter misses the rarer wrapping; match it after lines join.
_NICE_RIGHTS_NOTICE = re.compile(
    r"[©�]\s*NICE\s+\d{4}\.\s*All rights reserved\.\s*Subject to Notice of "
    r"rights\s*\([^)]*\)\.?"
)
# Bullet glyphs as they come out of the extractor; U+FFFD is a bullet whose
# font had no Unicode mapping.
_BULLET = re.compile(r"[●•▪◦‣⁃�]")
_WHITESPACE = re.compile(r"\s+")
_UNSAFE_FILE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass(frozen=True)
class Corpus:
    """A folder of PDFs and the vector store they are indexed into.

    ``folder`` is a callable so the location is read from settings at call time.
    NICE PDFs carry clean titles in their metadata ("Type 2 diabetes in adults:
    management") while their file names do not, so ``metadata_titles`` prefers
    the metadata.
    """

    name: str
    store: VectorStore
    folder: Callable[[], Path]
    metadata_titles: bool = False


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
    text = _NICE_RIGHTS_NOTICE.sub(" ", text)
    text = _DOT_LEADER.sub(" ", text)
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


def pdf_metadata_title(source: Path | BinaryIO) -> str | None:
    """Return the title recorded in a PDF's metadata, if it has a usable one."""
    from pypdf import PdfReader

    title = (PdfReader(source).metadata or {}).get("/Title")
    title = _WHITESPACE.sub(" ", str(title or "")).strip()
    return title or None


def _title_for(corpus: Corpus, path: Path, source: Path | BinaryIO) -> str:
    if corpus.metadata_titles:
        try:
            if title := pdf_metadata_title(source):
                return title
        except Exception:  # a damaged metadata block must not block indexing
            logger.warning("Could not read the metadata title of %s", path.name)
    return document_title(path)


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


def _indexed_pdf_ids(corpus: Corpus) -> set[str]:
    return {
        document_id
        for document_id in corpus.store.document_ids()
        if document_id.startswith(PDF_DOCUMENT_PREFIX)
    }


def _sync_unlocked(corpus: Corpus, directory: Path) -> dict[str, int]:
    wanted = _files_by_document_id(directory)
    if not wanted:
        logger.warning("No PDF documents found in %s", directory)
    indexed = _indexed_pdf_ids(corpus)

    removed = 0
    for document_id in indexed - wanted.keys():
        removed += 1
        corpus.store.remove_document(document_id)

    added = 0
    for document_id, path in wanted.items():
        if document_id in indexed:
            continue
        text = extract_pdf_text(path)
        if not text:
            logger.warning("No extractable text in %s; is it a scanned PDF?", path.name)
            continue
        result = index_into(
            corpus.store, _title_for(corpus, path, path), text, ["pdf"], document_id=document_id
        )
        added += 1
        logger.info("Indexed %s as %s chunks", path.name, result["chunk_count"])

    logger.info(
        "Document sync (%s): %d added, %d removed, %d unchanged",
        corpus.name,
        added,
        removed,
        len(wanted.keys() & indexed),
    )
    return {"added": added, "removed": removed}


def sync(corpus: Corpus, directory: Path | None = None) -> dict[str, int]:
    """Bring a corpus's store in line with the PDFs in its folder."""
    with _lock:
        return _sync_unlocked(corpus, directory or corpus.folder())


def rebuild(corpus: Corpus, directory: Path | None = None) -> dict[str, int]:
    """Drop every vector in a corpus's store and index its folder from scratch."""
    with _lock:
        corpus.store.reset()
        _sync_unlocked(corpus, directory or corpus.folder())
    return corpus.store.stats()


def upload_pdf(
    corpus: Corpus, file_name: str, content: bytes, directory: Path | None = None
) -> dict[str, object]:
    """Save an uploaded PDF into a corpus's folder and index it.

    An existing document is never replaced or removed. A different PDF that
    arrives under a name already in use is saved under a suffixed name and
    indexed alongside the first; only byte-identical content is rejected, since
    indexing it again would add nothing but duplicate chunks.
    """
    directory = directory or corpus.folder()
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
    title = _title_for(corpus, Path(safe_name), io.BytesIO(content))

    with _lock:
        existing = _files_by_document_id(directory)
        if document_id in existing or document_id in _indexed_pdf_ids(corpus):
            name = existing[document_id].name if document_id in existing else safe_name
            raise DuplicateDocumentError(
                f"This PDF is already in the {corpus.name} as '{name}'."
            )

        directory.mkdir(parents=True, exist_ok=True)
        target = _unused_path(directory, safe_name)
        temporary = target.with_suffix(".pdf.tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)

        result = index_into(corpus.store, title, text, ["pdf"], document_id=document_id)

    logger.info(
        "Uploaded %s to the %s as %s chunks", target.name, corpus.name, result["chunk_count"]
    )
    return {
        "document_id": document_id,
        "chunk_count": int(result["chunk_count"]),
        "file_name": target.name,
        "title": title,
    }


def remove_document(corpus: Corpus, document_id: str, directory: Path | None = None) -> int:
    """Remove a document's chunks and, for a PDF, its file. Returns chunks removed."""
    directory = directory or corpus.folder()
    with _lock:
        if document_id.startswith(PDF_DOCUMENT_PREFIX):
            path = _files_by_document_id(directory).get(document_id)
            if path is not None:
                path.unlink()
                logger.info("Deleted %s", path.name)
        return corpus.store.remove_document(document_id)
