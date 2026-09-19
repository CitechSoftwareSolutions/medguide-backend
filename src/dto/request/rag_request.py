"""Validated request DTOs for owner-managed knowledge indexing."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.validations import clean_tags, require_text


class IndexDocumentRequest(BaseModel):
    """Knowledge the owner wants searchable immediately."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        max_length=200,
        description="A short subject line; it is embedded with every chunk.",
    )
    text: str = Field(description="The knowledge body to chunk and index.")
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("title", "text")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Trim text and reject an empty value."""
        return require_text(value)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        """Normalize tags the same way knowledge entries do."""
        return clean_tags(value)
