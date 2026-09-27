"""Response DTOs for knowledge indexing endpoints."""

from pydantic import BaseModel, Field


class IndexDocumentData(BaseModel):
    """The result of indexing one document."""

    document_id: str = Field(description="Use this identifier to remove the document.")
    chunk_count: int


class IndexDocumentResponse(BaseModel):
    """The standard response body for an indexing request."""

    data: IndexDocumentData


class UploadDocumentData(BaseModel):
    """The result of uploading and indexing one PDF."""

    document_id: str = Field(description="Use this identifier to remove the document.")
    chunk_count: int
    file_name: str = Field(
        description="The name the PDF was saved under; suffixed _2, _3... if taken."
    )
    title: str = Field(description="The title embedded with every chunk.")


class UploadDocumentResponse(BaseModel):
    """The standard response body for a PDF upload."""

    data: UploadDocumentData


class DocumentSummaryData(BaseModel):
    """One document in the vector store."""

    document_id: str
    title: str
    tags: list[str]
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Every document currently searchable."""

    data: list[DocumentSummaryData]
    count: int


class RagStatsData(BaseModel):
    """How much knowledge is currently searchable."""

    document_count: int
    chunk_count: int


class RagStatsResponse(BaseModel):
    """The standard response body for index statistics."""

    data: RagStatsData
