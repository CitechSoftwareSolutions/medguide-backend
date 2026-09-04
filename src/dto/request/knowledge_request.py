"""Validated request DTOs for knowledge-entry endpoints."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.enums import KnowledgeType
from src.validations import clean_tags, require_text


class CreateKnowledgeEntryRequest(BaseModel):
    """Data required to create one medical knowledge entry."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="A unique, human-readable entry title.")
    summary: str = Field(description="A concise, clinically neutral explanation.")
    knowledge_type: KnowledgeType
    tags: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("title", "summary")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        """Trim text and reject an empty value."""
        return require_text(value)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        """Normalize tags before the service stores them."""
        return clean_tags(value)
