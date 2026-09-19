"""Response DTOs for knowledge indexing endpoints."""

from pydantic import BaseModel, Field


class IndexDocumentData(BaseModel):
    """The result of indexing one document."""

    document_id: str = Field(description="Use this identifier to remove the document.")
    chunk_count: int


class IndexDocumentResponse(BaseModel):
    """The standard response body for an indexing request."""

    data: IndexDocumentData


class RagStatsData(BaseModel):
    """How much knowledge is currently searchable."""

    document_count: int
    chunk_count: int


class RagStatsResponse(BaseModel):
    """The standard response body for index statistics."""

    data: RagStatsData
