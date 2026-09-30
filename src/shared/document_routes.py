"""The document endpoints every tool exposes, built once and mounted per tool.

Tool 01 mounts them at ``/api/v1/rag`` and Tool 02 at ``/api/v1/nice``; each
passes its own ``service`` module. A service provides ``index_document``,
``list_documents``, ``upload_pdf``, ``remove_document``, ``rebuild`` and
``stats``.

These endpoints change what the assistant treats as ground truth and are
unauthenticated. Put them behind an owner-only guard before this runs anywhere
but a development machine.
"""

from types import ModuleType

from fastapi import APIRouter, File, UploadFile, status

from src.shared.config import get_settings
from src.shared.errors import DocumentTooLargeError
from src.shared.schemas import (
    DocumentListResponse,
    DocumentSummaryData,
    IndexDocumentData,
    IndexDocumentRequest,
    IndexDocumentResponse,
    StoreStatsData,
    StoreStatsResponse,
    UploadDocumentData,
    UploadDocumentResponse,
)


def build_document_router(*, prefix: str, tag: str, service: ModuleType) -> APIRouter:
    """Create the list, index, upload, delete, reindex and stats endpoints."""
    router = APIRouter(prefix=prefix, tags=[tag])

    def stats_response() -> StoreStatsResponse:
        return StoreStatsResponse(data=StoreStatsData(**service.stats()))

    @router.post(
        "/documents",
        response_model=IndexDocumentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def index_document(payload: IndexDocumentRequest) -> IndexDocumentResponse:
        """Index text, searchable immediately without a restart."""
        result = service.index_document(payload.title, payload.text, payload.tags)
        return IndexDocumentResponse(
            data=IndexDocumentData(
                document_id=str(result["document_id"]),
                chunk_count=int(result["chunk_count"]),
            )
        )

    @router.get("/documents", response_model=DocumentListResponse)
    def list_documents() -> DocumentListResponse:
        """List every document in the store with its chunk count."""
        documents = [
            DocumentSummaryData.model_validate(item) for item in service.list_documents()
        ]
        return DocumentListResponse(data=documents, count=len(documents))

    @router.post(
        "/documents/upload",
        response_model=UploadDocumentResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def upload_document(
        file: UploadFile = File(description="A text-based PDF guideline."),
    ) -> UploadDocumentResponse:
        """Upload a PDF, save it to the tool's ``data`` folder and index it.

        A name already in use gets a ``_2``, ``_3``... suffix; nothing is replaced.
        """
        limit = get_settings().max_upload_bytes
        # Read one byte past the limit so an oversized body is detected without
        # buffering all of it.
        content = file.file.read(limit + 1)
        if len(content) > limit:
            raise DocumentTooLargeError(
                f"The file exceeds the {limit // (1024 * 1024)} MB upload limit."
            )
        result = service.upload_pdf(file.filename or "document.pdf", content)
        return UploadDocumentResponse(data=UploadDocumentData.model_validate(result))

    @router.delete("/documents/{document_id}", response_model=StoreStatsResponse)
    def remove_document(document_id: str) -> StoreStatsResponse:
        """Remove one document's chunks; for a PDF, delete the file too."""
        service.remove_document(document_id)
        return stats_response()

    @router.post("/reindex", response_model=StoreStatsResponse)
    def reindex() -> StoreStatsResponse:
        """Rebuild the store from the PDFs in the tool's ``data`` folder.

        Documents added through ``POST /documents`` are dropped by a rebuild.
        """
        service.rebuild()
        return stats_response()

    @router.get("/stats", response_model=StoreStatsResponse)
    def store_stats() -> StoreStatsResponse:
        """Report how much is currently searchable."""
        return stats_response()

    return router
