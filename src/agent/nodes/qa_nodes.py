"""Nodes of the knowledge question-answering subgraph.

Each node takes the subgraph state and returns only the keys it changes.

Two invariants hold across this module:

* Retrieval is deterministic. The model never decides *whether* to look
  something up, only how to phrase the query, which removes a large class of
  nondeterminism from the answer path.
* Every node that can fail degrades toward a usable path rather than raising,
  except answer synthesis itself. An answer is only ever emitted after
  ``verify_groundedness`` has passed it or the graph has abstained.
"""

from __future__ import annotations

import logging
import re

import anyio
from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm import complete_structured, complete_text
from src.agent.prompts import (
    ABSTENTION_MESSAGE,
    BROADEN_QUERY_SYSTEM,
    GRADE_RELEVANCE_SYSTEM,
    REWRITE_QUERY_SYSTEM,
    SYNTHESIZE_SYSTEM,
    VERIFY_GROUNDEDNESS_SYSTEM,
)
from src.agent.state import Citation, Confidence, QaState, RetrievedChunk, Turn
from src.exceptions import AgentUnavailableError
from src.services import rag_service

logger = logging.getLogger(__name__)

MAX_RETRIEVAL_ATTEMPTS = 2
MAX_SYNTHESIS_ATTEMPTS = 2

_CITATION_MARKER = re.compile(r"\[(\d+)\]")


class RewrittenQuery(BaseModel):
    """A standalone search query derived from a possibly elliptical question."""

    model_config = ConfigDict(extra="forbid")

    search_query: str = Field(description="The standalone search query.")


class PassageVerdict(BaseModel):
    """Whether one retrieved passage is worth showing to the answerer."""

    model_config = ConfigDict(extra="forbid")

    passage: int = Field(description="The 1-based number of the passage judged.")
    keep: bool = Field(description="True when the passage helps answer the question.")
    reason: str = Field(description="A brief justification.")


class RelevanceGrade(BaseModel):
    """Verdicts for every retrieved passage."""

    model_config = ConfigDict(extra="forbid")

    verdicts: list[PassageVerdict]


class GroundednessVerdict(BaseModel):
    """Whether a draft answer is fully supported by its passages."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool = Field(description="True when every clinical claim is supported.")
    # No default: a strict JSON schema requires every property to be listed as
    # required, and a Pydantic default would drop it from that list.
    unsupported_claims: list[str] = Field(
        description="Claims from the answer that no passage supports; empty when grounded.",
    )


def _format_history(history: list[Turn]) -> str:
    """Render recent turns for the query rewriter."""
    if not history:
        return "(no earlier conversation)"
    return "\n".join(
        f"{'Clinician' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in history
    )


def _format_passages(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered passages the model can cite by number."""
    return "\n\n".join(
        f"[{number}] {chunk['title']}\n{chunk['text']}"
        for number, chunk in enumerate(chunks, start=1)
    )


def _confidence(chunks: list[RetrievedChunk]) -> Confidence:
    """Grade confidence from the strength of the supporting matches."""
    if not chunks:
        return "low"
    best = max(chunk["score"] for chunk in chunks)
    if best >= 0.60:
        return "high"
    if best >= 0.45:
        return "medium"
    return "low"


def _citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve the bracketed markers the answer actually used.

    Only markers present in the answer become citations, so the response never
    claims provenance the text did not rely on. Out-of-range markers are dropped
    rather than trusted.
    """
    seen: list[int] = []
    for raw in _CITATION_MARKER.findall(answer):
        marker = int(raw)
        if 1 <= marker <= len(chunks) and marker not in seen:
            seen.append(marker)

    return [
        Citation(
            marker=marker,
            chunk_id=chunks[marker - 1]["chunk_id"],
            document_id=chunks[marker - 1]["document_id"],
            title=chunks[marker - 1]["title"],
            score=chunks[marker - 1]["score"],
        )
        for marker in sorted(seen)
    ]


async def rewrite_query(state: QaState) -> dict:
    """Turn the question plus history into a standalone search query."""
    question = state["question"]
    try:
        result = await complete_structured(
            system=REWRITE_QUERY_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Earlier conversation:\n{_format_history(state.get('history', []))}"
                        f"\n\nCurrent question:\n{question}"
                    ),
                }
            ],
            schema=RewrittenQuery,
            label="rewrite_query",
        )
        search_query = result.search_query.strip() or question
    except AgentUnavailableError:
        # Rewriting is an accuracy aid, not a requirement. Searching the raw
        # question is strictly better than failing the whole request here.
        logger.warning("Query rewrite failed; searching the raw question")
        search_query = question

    return {"search_query": search_query}


async def broaden_query(state: QaState) -> dict:
    """Widen a query by one step after a fruitless search."""
    current = state.get("search_query") or state["question"]
    try:
        result = await complete_structured(
            system=BROADEN_QUERY_SYSTEM,
            messages=[{"role": "user", "content": f"Query that found nothing:\n{current}"}],
            schema=RewrittenQuery,
            label="broaden_query",
        )
        broadened = result.search_query.strip() or current
    except AgentUnavailableError:
        logger.warning("Query broadening failed; reusing the previous query")
        broadened = current

    return {"search_query": broadened}


async def retrieve(state: QaState) -> dict:
    """Search the vector store. Deterministic; no model involved.

    The embedding call is CPU-bound and blocking, so it runs on a worker thread
    to keep the event loop free.
    """
    query = state.get("search_query") or state["question"]
    chunks = await anyio.to_thread.run_sync(rag_service.retrieve, query)
    logger.debug("Retrieved %d chunks for %r", len(chunks), query)
    return {
        "chunks": chunks,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


async def grade_relevance(state: QaState) -> dict:
    """Drop retrieved passages that merely share vocabulary with the question."""
    chunks = state.get("chunks", [])
    if not chunks:
        return {"relevant_chunks": []}

    try:
        grade = await complete_structured(
            system=GRADE_RELEVANCE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{state['question']}\n\n"
                        f"Passages:\n{_format_passages(chunks)}"
                    ),
                }
            ],
            schema=RelevanceGrade,
            label="grade_relevance",
        )
    except AgentUnavailableError:
        # Keep what the score floor already allowed. The groundedness check
        # downstream is the real accuracy gate, so a grader outage costs
        # precision, not safety.
        logger.warning("Relevance grading failed; keeping all retrieved passages")
        return {"relevant_chunks": chunks}

    keep = {
        verdict.passage
        for verdict in grade.verdicts
        if verdict.keep and 1 <= verdict.passage <= len(chunks)
    }
    relevant = [chunk for number, chunk in enumerate(chunks, start=1) if number in keep]
    logger.debug("Kept %d of %d passages after grading", len(relevant), len(chunks))
    return {"relevant_chunks": relevant}


async def synthesize(state: QaState) -> dict:
    """Write the answer from the graded passages, citing each claim."""
    chunks = state.get("relevant_chunks", [])
    attempts = state.get("synthesis_attempts", 0)

    content = (
        f"Question:\n{state['question']}\n\n"
        f"Passages:\n{_format_passages(chunks)}"
    )
    if unsupported := state.get("unsupported_claims"):
        # A retry: name what failed so the second attempt does not repeat it.
        content += (
            "\n\nYour previous answer was rejected because these claims were not "
            "supported by the passages:\n"
            + "\n".join(f"- {claim}" for claim in unsupported)
            + "\n\nRewrite the answer using only what the passages state. Omit any "
            "claim you cannot cite."
        )

    answer = await complete_text(
        system=SYNTHESIZE_SYSTEM,
        messages=[{"role": "user", "content": content}],
        label="synthesize",
        effort="high",
    )

    return {
        "answer": answer,
        "citations": _citations(answer, chunks),
        "synthesis_attempts": attempts + 1,
    }


async def verify_groundedness(state: QaState) -> dict:
    """Audit the draft answer against its passages before it can be returned."""
    chunks = state.get("relevant_chunks", [])
    answer = state.get("answer", "")

    try:
        verdict = await complete_structured(
            system=VERIFY_GROUNDEDNESS_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Passages:\n{_format_passages(chunks)}\n\n"
                        f"Draft answer:\n{answer}"
                    ),
                }
            ],
            schema=GroundednessVerdict,
            label="verify_groundedness",
        )
    except AgentUnavailableError:
        # Fail closed. An unverified answer is exactly what this graph exists to
        # avoid emitting, so an outage here abstains instead of passing it.
        logger.warning("Groundedness check failed; abstaining rather than answering")
        return {"grounded": False, "unsupported_claims": []}

    if not verdict.grounded:
        logger.info(
            "Answer rejected as ungrounded (%d unsupported claims)",
            len(verdict.unsupported_claims),
        )

    return {
        "grounded": verdict.grounded,
        "unsupported_claims": verdict.unsupported_claims,
        "confidence": _confidence(chunks) if verdict.grounded else "abstained",
    }


async def abstain(state: QaState) -> dict:
    """Decline to answer. Deterministic; no model involved."""
    query = state.get("search_query") or state["question"]
    return {
        "answer": ABSTENTION_MESSAGE.format(query=query),
        "citations": [],
        "confidence": "abstained",
        "abstained": True,
    }


def after_grade(state: QaState) -> str:
    """Route on whether anything relevant was found, with one retry allowed."""
    if state.get("relevant_chunks"):
        return "synthesize"
    if state.get("retrieval_attempts", 0) < MAX_RETRIEVAL_ATTEMPTS:
        return "broaden_query"
    return "abstain"


def after_verify(state: QaState) -> str:
    """Route on the groundedness verdict, with one rewrite allowed."""
    if state.get("grounded"):
        return "finish"
    if state.get("synthesis_attempts", 0) < MAX_SYNTHESIS_ATTEMPTS:
        return "synthesize"
    return "abstain"
