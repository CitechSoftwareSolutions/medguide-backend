"""HTTP endpoints for owner-managed knowledge indexing.

These endpoints mutate what the assistant will treat as ground truth and are
currently unauthenticated. Put them behind an owner-only guard before this runs
anywhere but a development machine.
"""

from fastapi import APIRouter, File, UploadFile, status

from src.config import get_settings
from src.controllers import (
    index_document,
    list_documents,
    reindex,
    remove_document,
    stats,
    upload_document,
)
from src.dto import (
    DocumentListResponse,
    IndexDocumentRequest,
    IndexDocumentResponse,
    RagStatsResponse,
    UploadDocumentResponse,
)
from src.exceptions import DocumentTooLargeError

router = APIRouter(prefix="/api/v1/rag", tags=["knowledge-index"])


@router.post(
    "/documents",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def index_knowledge_document(payload: IndexDocumentRequest) -> IndexDocumentResponse:
    """Index new knowledge, searchable immediately without a restart."""
    return index_document(payload)


@router.get("/documents", response_model=DocumentListResponse)
def list_knowledge_documents() -> DocumentListResponse:
    """List every document in the vector store with its chunk count."""
    return list_documents()


@router.post(
    "/documents/upload",
    response_model=UploadDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_knowledge_document(
    file: UploadFile = File(description="A text-based PDF guideline."),
) -> UploadDocumentResponse:
    """Upload a PDF, save it to ``src/data`` and index it immediately.

    Uploading a file with the same name as an existing PDF replaces it.
    """
    limit = get_settings().max_upload_bytes
    # Read one byte past the limit so an oversized body is detected without
    # buffering all of it.
    content = file.file.read(limit + 1)
    if len(content) > limit:
        raise DocumentTooLargeError(
            f"The file exceeds the {limit // (1024 * 1024)} MB upload limit."
        )
    return upload_document(file.filename or "document.pdf", content)


@router.delete("/documents/{document_id}", response_model=RagStatsResponse)
def remove_knowledge_document(document_id: str) -> RagStatsResponse:
    """Remove one document's chunks; for an uploaded PDF, delete the file too."""
    return remove_document(document_id)


@router.post("/reindex", response_model=RagStatsResponse)
def reindex_knowledge_base() -> RagStatsResponse:
    """Rebuild the whole index from the PDFs in ``src/data``.

    Documents added through ``POST /documents`` are dropped by a rebuild.
    """
    return reindex()


@router.get("/stats", response_model=RagStatsResponse)
def knowledge_index_stats() -> RagStatsResponse:
    """Report how much knowledge is currently searchable."""
    return stats()
