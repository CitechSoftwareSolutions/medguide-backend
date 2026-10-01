"""The graph nodes Tool 02 adds to the agent's knowledge_qa graph.

* ``retrieve_nice`` searches NICE guidance in the same step as Tool 01's search.
* ``validate_against_nice`` checks the finished, grounded answer against those
  passages and reports ``nice_validation``. It never changes the answer.

Both fail open: the answer has already been verified against the knowledge
base, so a NICE outage reports ``unavailable`` instead of withholding it.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field

from src.agent.citations import format_passages, resolve_markers, strip_markers
from src.agent.ollama_client import complete_structured
from src.agent.state import (
    NiceConflict,
    NiceValidation,
    QaState,
    RetrievedChunk,
    nice_validation,
)
from src.shared.errors import AgentUnavailableError
from src.tools.tool_02_nice_guidance import service
from src.tools.tool_02_nice_guidance.prompts import NICE_VALIDATION_SYSTEM

logger = logging.getLogger(__name__)


class NiceClaimCheck(BaseModel):
    """How the NICE passages bear on one claim in the answer."""

    model_config = ConfigDict(extra="forbid")

    claim: str = Field(description="The answer's claim, quoted verbatim.")
    verdict: Literal["agrees", "contradicts", "not_addressed"]
    nice_says: str = Field(
        description="What the NICE passages say on this point; empty when not_addressed."
    )
    passages: list[int] = Field(
        description="The 1-based numbers of the NICE passages that say it; empty when "
        "not_addressed."
    )


class NiceVerdict(BaseModel):
    """A verdict for every clinical claim in the answer.

    The model judges one claim at a time; the overall status is derived in code
    by :func:`_nice_result`, which a small model does far less reliably.
    """

    model_config = ConfigDict(extra="forbid")

    claims: list[NiceClaimCheck]


async def retrieve_nice(state: QaState) -> dict:
    """Search NICE guidance with the same query Tool 01 is using."""
    query = state.get("search_query") or state["question"]
    try:
        chunks = await anyio.to_thread.run_sync(service.retrieve, query)
    except Exception:
        logger.exception("NICE search failed; the answer will not be validated")
        return {"nice_chunks": [], "nice_search_failed": True}
    logger.debug("Retrieved %d NICE chunks for %r", len(chunks), query)
    return {"nice_chunks": chunks, "nice_search_failed": False}


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


_WORD = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")
_STOPWORDS = frozenset("a an and are as at be by for from in is of on or the to with".split())
# Share of each side's words the other must contain to count as a restatement.
_RESTATEMENT_OVERLAP = 0.85


def _words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _restates(claim: str, nice_says: str) -> bool:
    """Whether "what NICE says" is, in substance, the claim itself.

    A small model sometimes marks a claim as contradicted while quoting the claim
    back as NICE's position. Checked in both directions, so "Offer aspirin"
    against "Do not offer aspirin" is still a contradiction, as is any change
    of number.
    """
    claim_words, nice_words = _words(claim), _words(nice_says)
    if not claim_words or not nice_words:
        return False
    shared = len(claim_words & nice_words)
    return (
        shared / len(claim_words) >= _RESTATEMENT_OVERLAP
        and shared / len(nice_words) >= _RESTATEMENT_OVERLAP
    )


def _nice_result(verdict: NiceVerdict, chunks: list[RetrievedChunk]) -> NiceValidation:
    """Derive the overall result from the per-claim verdicts.

    A verdict counts only when it is backed: it names at least one NICE passage
    that exists and, for a contradiction, states what NICE says instead. An
    unbacked "agrees" or "contradicts" is treated as not addressed, which stops
    a claim NICE never mentions from being reported as a conflict. A
    "contradiction" whose NICE position restates the claim counts as agreement.

    The summary is written here rather than by the model, which tends to copy
    passage text instead of summarizing.
    """
    agrees: list[list[int]] = []
    conflicts: list[NiceConflict] = []
    not_addressed = 0
    for check in verdict.claims:
        claim = check.claim.strip()
        if not claim:
            continue
        markers = [citation["marker"] for citation in resolve_markers(check.passages, chunks)]
        nice_says = check.nice_says.strip()
        verdict_label = check.verdict
        if verdict_label == "contradicts" and _restates(claim, nice_says):
            verdict_label = "agrees"
        if verdict_label == "contradicts" and markers and nice_says:
            conflicts.append(NiceConflict(claim=claim, nice_position=nice_says, markers=markers))
        elif verdict_label == "agrees" and markers:
            agrees.append(markers)
        else:
            not_addressed += 1

    unaddressed = (
        f" {_plural(not_addressed, 'claim')} not addressed by NICE." if not_addressed else ""
    )
    if conflicts:
        status = "conflicts"
        summary = (
            f"NICE guidance disagrees with {_plural(len(conflicts), 'claim')} in this answer."
        )
    elif agrees:
        status = "consistent"
        summary = (
            f"NICE guidance supports {_plural(len(agrees), 'claim')} in this answer "
            f"and contradicts none.{unaddressed}"
        )
    else:
        status = "not_covered"
        summary = "The NICE passages found do not address the claims in this answer."

    cited = [marker for markers in agrees for marker in markers]
    cited += [marker for conflict in conflicts for marker in conflict["markers"]]
    return NiceValidation(
        status=status,
        summary=summary,
        conflicts=conflicts,
        citations=resolve_markers(cited, chunks),
    )


async def validate_against_nice(state: QaState) -> dict:
    """Check the grounded answer against the NICE passages, without changing it."""
    if state.get("nice_search_failed"):
        return {
            "nice_validation": nice_validation(
                "unavailable", "NICE guidance could not be searched for this question."
            )
        }

    chunks = state.get("nice_chunks", [])
    if not chunks:
        return {
            "nice_validation": nice_validation(
                "not_covered", "No NICE guidance passage matched this question."
            )
        }

    # The answer's [n] markers point at knowledge base passages the checker
    # never sees; left in, they would read as references to NICE passages.
    answer = strip_markers(state.get("answer", ""))
    try:
        verdict = await complete_structured(
            system=NICE_VALIDATION_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{state['question']}\n\n"
                        f"Answer to check:\n{answer}\n\n"
                        f"NICE passages:\n{format_passages(chunks)}"
                    ),
                }
            ],
            schema=NiceVerdict,
            label="validate_against_nice",
        )
    except AgentUnavailableError:
        logger.warning("NICE validation failed; returning the answer unchecked")
        return {
            "nice_validation": nice_validation(
                "unavailable", "The answer could not be checked against NICE guidance."
            )
        }

    result = _nice_result(verdict, chunks)
    logger.info("NICE validation: %s (%d conflicts)", result["status"], len(result["conflicts"]))
    return {"nice_validation": result}
