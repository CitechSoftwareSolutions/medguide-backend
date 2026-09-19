"""HTTP endpoints for owner-managed knowledge indexing.

These endpoints mutate what the assistant will treat as ground truth and are
currently unauthenticated. Put them behind an owner-only guard before this runs
anywhere but a development machine.
"""

from fastapi import APIRouter, status

from src.controllers import index_document, reindex, remove_document, stats
from src.dto import IndexDocumentRequest, IndexDocumentResponse, RagStatsResponse

router = APIRouter(prefix="/api/v1/rag", tags=["knowledge-index"])


@router.post(
    "/documents",
    response_model=IndexDocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def index_knowledge_document(payload: IndexDocumentRequest) -> IndexDocumentResponse:
    """Index new knowledge, searchable immediately without a restart."""
    return index_document(payload)


@router.delete("/documents/{document_id}", response_model=RagStatsResponse)
def remove_knowledge_document(document_id: str) -> RagStatsResponse:
    """Remove one document's chunks from the index."""
    return remove_document(document_id)


@router.post("/reindex", response_model=RagStatsResponse)
def reindex_knowledge_base() -> RagStatsResponse:
    """Rebuild the whole index from the persisted knowledge entries."""
    return reindex()


@router.get("/stats", response_model=RagStatsResponse)
def knowledge_index_stats() -> RagStatsResponse:
    """Report how much knowledge is currently searchable."""
    return stats()
