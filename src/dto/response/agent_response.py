"""Response DTOs for the question-answering endpoint."""

from typing import Literal

from pydantic import BaseModel, Field


class CitationData(BaseModel):
    """Where one bracketed marker in the answer came from."""

    marker: int = Field(description="The [n] marker used in the answer text.")
    chunk_id: str
    document_id: str
    title: str
    score: float = Field(description="Cosine similarity of the supporting chunk.")


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
    route: str = Field(description="The capability that handled the request.")


class AnswerResponse(BaseModel):
    """The standard response body for one answer."""

    data: AnswerData
