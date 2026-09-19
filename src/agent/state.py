"""Typed state shared by the supervisor graph and its capability subgraphs."""

from typing import Annotated, Literal, NotRequired, TypedDict

from src.config import get_settings

Role = Literal["user", "assistant"]
Confidence = Literal["high", "medium", "low", "abstained"]


class Turn(TypedDict):
    """One recorded exchange line in a chat session."""

    role: Role
    content: str


class RetrievedChunk(TypedDict):
    """A knowledge chunk returned by vector search."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    score: float


class Citation(TypedDict):
    """The provenance of one claim in a synthesized answer."""

    marker: int
    chunk_id: str
    document_id: str
    title: str
    score: float


def append_turns(existing: list[Turn] | None, incoming: list[Turn]) -> list[Turn]:
    """Append new turns and keep only the most recent window.

    Trimming happens in the reducer rather than at prompt-build time because the
    checkpointer never evicts on its own; an unbounded history would grow the
    stored checkpoint for the lifetime of the process.
    """
    combined = [*(existing or []), *incoming]
    max_turns = get_settings().session_max_turns
    return combined[-max_turns:] if max_turns > 0 else combined


class SupervisorState(TypedDict):
    """State owned by the routing graph."""

    session_id: str
    question: str
    history: Annotated[list[Turn], append_turns]
    route: NotRequired[str]
    answer: NotRequired[str]
    citations: NotRequired[list[Citation]]
    confidence: NotRequired[Confidence]


class QaState(TypedDict):
    """State owned by the knowledge question-answering subgraph.

    The ``session_id``, ``question``, ``history``, ``answer``, ``citations`` and
    ``confidence`` keys are named to match :class:`SupervisorState` so values
    cross the subgraph boundary without an explicit adapter. ``history`` carries
    no reducer here: the subgraph only reads it, and declaring the reducer twice
    would append the same turns at both levels.
    """

    session_id: str
    question: str
    history: list[Turn]
    search_query: NotRequired[str]
    chunks: NotRequired[list[RetrievedChunk]]
    relevant_chunks: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[str]
    citations: NotRequired[list[Citation]]
    confidence: NotRequired[Confidence]
    grounded: NotRequired[bool]
    unsupported_claims: NotRequired[list[str]]
    retrieval_attempts: NotRequired[int]
    synthesis_attempts: NotRequired[int]
    abstained: NotRequired[bool]
