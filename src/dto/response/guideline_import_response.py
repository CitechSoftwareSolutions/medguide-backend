"""Response DTOs for the guideline bulk-import endpoint."""

from uuid import UUID

from pydantic import BaseModel


class ImportedEntryData(BaseModel):
    """One knowledge entry produced (or reused) by an import."""

    id: UUID
    title: str
    chunk_count: int
    created: bool


class ImportGuidelineData(BaseModel):
    """Everything one guideline import produced."""

    entries: list[ImportedEntryData]
    total_chunk_count: int


class ImportGuidelineResponse(BaseModel):
    """The standard response body for a guideline import."""

    data: ImportGuidelineData
