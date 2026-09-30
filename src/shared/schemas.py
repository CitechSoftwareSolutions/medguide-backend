"""Request and response models for the document endpoints of both tools."""

from pydantic import BaseModel, ConfigDict, Field, field_validator


def require_text(value: str) -> str:
    """Return trimmed text, rejecting empty or whitespace-only values."""
    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError("This field must contain text.")
    return cleaned_value


def clean_tags(tags: list[str]) -> list[str]:
    """Trim and lower-case tags, remove duplicates, and reject empty tags."""
    cleaned_tags = [require_text(tag).lower() for tag in tags]
    return list(dict.fromkeys(cleaned_tags))


class IndexDocumentRequest(BaseModel):
    """Text to make searchable immediately."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        max_length=200,
        description="A short subject line; it is embedded with every chunk.",
    )
    text: str = Field(description="The body to chunk and index.")
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("title", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Trim text and reject an empty value."""
        return require_text(value)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        """Normalize tags."""
        return clean_tags(value)


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
    """One document in a vector store."""

    document_id: str
    title: str
    tags: list[str]
    chunk_count: int


class DocumentListResponse(BaseModel):
    """Every document currently searchable."""

    data: list[DocumentSummaryData]
    count: int


class StoreStatsData(BaseModel):
    """How much is currently searchable."""

    document_count: int
    chunk_count: int


class StoreStatsResponse(BaseModel):
    """The standard response body for store statistics."""

    data: StoreStatsData
