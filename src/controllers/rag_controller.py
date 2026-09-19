"""Functions that shape indexing results into API response DTOs."""

from src.dto import (
    IndexDocumentData,
    IndexDocumentRequest,
    IndexDocumentResponse,
    RagStatsData,
    RagStatsResponse,
)
from src.services import rag_service


def index_document(payload: IndexDocumentRequest) -> IndexDocumentResponse:
    """Chunk, embed and index one document so it is searchable immediately."""
    result = rag_service.index_document(payload.title, payload.text, payload.tags)
    return IndexDocumentResponse(
        data=IndexDocumentData(
            document_id=str(result["document_id"]),
            chunk_count=int(result["chunk_count"]),
        )
    )


def remove_document(document_id: str) -> RagStatsResponse:
    """Remove one document's chunks and report the resulting index size."""
    rag_service.remove_document(document_id)
    return stats()


def reindex() -> RagStatsResponse:
    """Rebuild the index from the persisted knowledge entries."""
    rag_service.reindex_from_knowledge_entries()
    return stats()


def stats() -> RagStatsResponse:
    """Report how much knowledge is searchable."""
    current = rag_service.stats()
    return RagStatsResponse(
        data=RagStatsData(
            document_count=current["document_count"],
            chunk_count=current["chunk_count"],
        )
    )
