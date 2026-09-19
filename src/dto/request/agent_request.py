"""Validated request DTOs for the question-answering endpoint."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.validations import require_text


class AskQuestionRequest(BaseModel):
    """One clinical question, optionally continuing an existing chat session."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        max_length=2000,
        description="The clinical question to answer from the knowledge base.",
    )
    session_id: UUID | None = Field(
        default=None,
        description=(
            "Return the value from a previous answer to continue that "
            "conversation. Omit it to start a new session."
        ),
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Trim the question and reject an empty one."""
        return require_text(value)
