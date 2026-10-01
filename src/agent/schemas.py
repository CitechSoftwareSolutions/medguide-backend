"""Request and response models for ``POST /api/v1/agent/ask``."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.shared.schemas import require_text


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


class CitationData(BaseModel):
    """Where one bracketed marker in the answer came from."""

    marker: int = Field(description="The [n] marker used in the answer text.")
    chunk_id: str
    document_id: str
    title: str
    score: float = Field(description="Cosine similarity of the supporting chunk.")


class NiceConflictData(BaseModel):
    """One claim in the answer that NICE guidance disagrees with."""

    claim: str = Field(description="The answer's claim, as quoted by the checker.")
    nice_position: str = Field(description="What NICE guidance says instead.")
    markers: list[int] = Field(
        description="Markers into nice_validation.citations for the NICE passages."
    )


class NiceValidationData(BaseModel):
    """How the answer compares with NICE guidance. It never changes the answer."""

    status: Literal["consistent", "conflicts", "not_covered", "unavailable", "not_run"] = (
        Field(
            description=(
                "'consistent': NICE agrees. 'conflicts': NICE disagrees on the listed "
                "points. 'not_covered': no NICE guidance addresses the answer. "
                "'unavailable': the check could not run. 'not_run': there was no "
                "answer to check."
            )
        )
    )
    summary: str
    conflicts: list[NiceConflictData] = Field(default_factory=list)
    citations: list[CitationData] = Field(
        default_factory=list, description="NICE passages referenced by [n] markers."
    )


def _nice_not_run() -> NiceValidationData:
    return NiceValidationData(
        status="not_run",
        summary="No answer was given, so nothing was checked against NICE.",
    )


class StepTimingData(BaseModel):
    """How long one step took. Model fields appear only for steps that called MedGemma."""

    step: str = Field(description="The step; tool steps start with tool_01. or tool_02.")
    seconds: float
    model_calls: int | None = None
    prompt_tokens: int | None = Field(default=None, description="Tokens MedGemma read.")
    prompt_seconds: float | None = None
    output_tokens: int | None = Field(
        default=None, description="Tokens MedGemma wrote, including its thinking."
    )
    output_seconds: float | None = None


class AnswerData(BaseModel):
    """One answer, with the provenance the client needs to verify it."""

    session_id: str = Field(
        description="Pass this back on the next question to continue the session."
    )
    answer: str
    citations: list[CitationData] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "abstained"] = Field(
        description=(
            "'abstained' means the knowledge base did not support an answer and "
            "none was invented."
        )
    )
    nice_validation: NiceValidationData = Field(default_factory=_nice_not_run)
    timings: list[StepTimingData] = Field(
        default_factory=list, description="Each step in the order it finished."
    )
    total_seconds: float = Field(default=0.0, description="Time spent answering, in seconds.")


class AnswerResponse(BaseModel):
    """The standard response body for one answer."""

    data: AnswerData
