"""Nodes of the supervisor graph: routing, refusal and turn recording."""

from __future__ import annotations

import logging

from pydantic import BaseModel, ConfigDict, Field

from src.agent.llm import complete_structured
from src.agent.prompts import OUT_OF_SCOPE_MESSAGE, ROUTE_SYSTEM
from src.agent.state import SupervisorState, Turn
from src.agent.tools import registry
from src.exceptions import AgentUnavailableError

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """Which capability should handle a request."""

    model_config = ConfigDict(extra="forbid")

    capability: str = Field(description="The name of the chosen capability.")
    reason: str = Field(description="A brief justification.")


async def route(state: SupervisorState) -> dict:
    """Choose the capability that handles this question.

    Degrades to the knowledge lookup on any failure or unrecognised label. A
    router problem must never be the reason a clinician gets an error instead of
    an attempt at their answer.
    """
    try:
        decision = await complete_structured(
            system=ROUTE_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Available capabilities:\n{registry.render_menu()}\n\n"
                        f"Request:\n{state['question']}"
                    ),
                }
            ],
            schema=RouteDecision,
            label="route",
        )
        chosen = decision.capability.strip()
    except AgentUnavailableError:
        logger.warning("Routing failed; defaulting to %s", registry.KNOWLEDGE_QA)
        chosen = registry.KNOWLEDGE_QA

    if chosen not in registry.tool_names():
        logger.warning(
            "Router returned unknown capability %r; defaulting to %s",
            chosen,
            registry.KNOWLEDGE_QA,
        )
        chosen = registry.KNOWLEDGE_QA

    return {"route": chosen}


async def decline_out_of_scope(state: SupervisorState) -> dict:
    """Refuse a request no capability covers. Deterministic; no model involved."""
    return {
        "answer": OUT_OF_SCOPE_MESSAGE,
        "citations": [],
        "confidence": "abstained",
    }


async def record_turn(state: SupervisorState) -> dict:
    """Commit this exchange to session history.

    The state reducer handles appending and trimming, so this node only needs to
    hand over the two new turns.
    """
    return {
        "history": [
            Turn(role="user", content=state["question"]),
            Turn(role="assistant", content=state.get("answer", "")),
        ]
    }


def after_route(state: SupervisorState) -> str:
    """Send the request to the capability the router chose."""
    chosen = state.get("route", registry.KNOWLEDGE_QA)
    if chosen == registry.OUT_OF_SCOPE:
        return "decline_out_of_scope"
    return chosen
