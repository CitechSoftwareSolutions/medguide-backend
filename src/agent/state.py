"""Typed state shared by the supervisor graph and its capability subgraphs."""

from typing import Annotated, Literal, NotRequired, TypedDict

from src.shared.config import get_settings
from src.shared.indexing import RetrievedChunk

Role = Literal["user", "assistant"]
Confidence = Literal["high", "medium", "low", "abstained"]


class Turn(TypedDict):
    """One recorded exchange line in a chat session."""

    role: Role
    content: str


class Citation(TypedDict):
    """The provenance of one claim in a synthesized answer."""

    marker: int
    chunk_id: str
    document_id: str
    title: str
    score: float


NiceStatus = Literal["consistent", "conflicts", "not_covered", "unavailable", "not_run"]


class NiceConflict(TypedDict):
    """One claim in the answer that NICE guidance disagrees with."""

    claim: str
    nice_position: str
    # Markers into the NICE passages, resolved in NiceValidation.citations.
    markers: list[int]


class NiceValidation(TypedDict):
    """The result of checking an answer against NICE guidance (tool 02).

    It is reported next to the answer and never changes it.
    """

    status: NiceStatus
    summary: str
    conflicts: list[NiceConflict]
    citations: list[Citation]


def nice_validation(status: NiceStatus, summary: str) -> NiceValidation:
    """Build a result that carries no conflicts or citations."""
    return NiceValidation(status=status, summary=summary, conflicts=[], citations=[])


def nice_not_run() -> NiceValidation:
    """The result for a request that produced no answer to check."""
    return nice_validation("not_run", "No answer was given, so nothing was checked against NICE.")


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
    nice_validation: NotRequired[NiceValidation]


class QaState(TypedDict):
    """State owned by the knowledge question-answering subgraph.

    The ``session_id``, ``question``, ``history``, ``answer``, ``citations`` and
    ``confidence`` keys are named to match :class:`SupervisorState` so values
    cross the subgraph boundary without an explicit adapter. ``history`` carries
    no reducer here: the subgraph only reads it, and declaring the reducer twice
    would append the same turns at both levels.

    ``chunks`` (tool 01) and ``nice_chunks`` (tool 02) are written by different
    nodes, which is what lets the two retrievers run in the same step.
    ``nice_chunks`` is read only by ``validate_against_nice``; no node that
    writes the answer ever sees it.
    """

    session_id: str
    question: str
    history: list[Turn]
    search_query: NotRequired[str]
    chunks: NotRequired[list[RetrievedChunk]]
    nice_chunks: NotRequired[list[RetrievedChunk]]
    nice_search_failed: NotRequired[bool]
    nice_validation: NotRequired[NiceValidation]
    relevant_chunks: NotRequired[list[RetrievedChunk]]
    answer: NotRequired[str]
    citations: NotRequired[list[Citation]]
    confidence: NotRequired[Confidence]
    grounded: NotRequired[bool]
    unsupported_claims: NotRequired[list[str]]
    retrieval_attempts: NotRequired[int]
    synthesis_attempts: NotRequired[int]
    abstained: NotRequired[bool]
