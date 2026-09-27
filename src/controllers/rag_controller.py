"""Functions that shape indexing results into API response DTOs."""

from src.dto import (
    DocumentListResponse,
    DocumentSummaryData,
    IndexDocumentData,
    IndexDocumentRequest,
    IndexDocumentResponse,
    RagStatsData,
    RagStatsResponse,
    UploadDocumentData,
    UploadDocumentResponse,
)
from src.repositories import vector_repository
from src.services import document_service, rag_service


def index_document(payload: IndexDocumentRequest) -> IndexDocumentResponse:
    """Chunk, embed and index one document so it is searchable immediately."""
    result = rag_service.index_document(payload.title, payload.text, payload.tags)
    return IndexDocumentResponse(
        data=IndexDocumentData(
            document_id=str(result["document_id"]),
            chunk_count=int(result["chunk_count"]),
        )
    )


def upload_document(file_name: str, content: bytes) -> UploadDocumentResponse:
    """Save an uploaded PDF into the documents directory and index it."""
    result = document_service.upload_pdf(file_name, content)
    return UploadDocumentResponse(data=UploadDocumentData.model_validate(result))


def list_documents() -> DocumentListResponse:
    """List every document in the vector store."""
    documents = [
        DocumentSummaryData.model_validate(item)
        for item in vector_repository.list_documents()
    ]
    return DocumentListResponse(data=documents, count=len(documents))


def remove_document(document_id: str) -> RagStatsResponse:
    """Remove one document's chunks (and PDF file) and report the index size."""
    document_service.remove_document(document_id)
    return stats()


def reindex() -> RagStatsResponse:
    """Rebuild the index from the PDFs in the documents directory."""
    document_service.rebuild_from_documents_directory()
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
