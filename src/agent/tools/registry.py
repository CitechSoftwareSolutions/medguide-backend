"""The catalogue of capabilities the supervisor can route to.

This module holds plain data only, no graph builders, so the router can read the
menu without importing any subgraph and creating an import cycle.

**Adding a capability is three edits:** append its spec here, write its subgraph
beside ``knowledge_qa_tool``, then add one node and one conditional edge in
``src/agent/graph.py``. Nothing in the existing question-answering path changes.
"""

from typing import TypedDict

KNOWLEDGE_QA = "knowledge_qa"
OUT_OF_SCOPE = "out_of_scope"


class ToolSpec(TypedDict):
    """How one capability advertises itself to the router."""

    name: str
    description: str


KNOWLEDGE_QA_SPEC: ToolSpec = {
    "name": KNOWLEDGE_QA,
    "description": (
        "Answer a clinical question using the institution's curated medical "
        "knowledge base: conditions, medications, procedures and symptoms, "
        "including dosing, contraindications and interactions where the entries "
        "record them. Use this for any request for medical information."
    ),
}

TOOL_SPECS: list[ToolSpec] = [KNOWLEDGE_QA_SPEC]


def tool_names() -> list[str]:
    """Return every routable capability name, including the refusal route."""
    return [spec["name"] for spec in TOOL_SPECS] + [OUT_OF_SCOPE]


def render_menu() -> str:
    """Render the capability menu for the routing prompt.

    Rendered into the router's *message* rather than its system prompt so adding
    a capability does not invalidate the cached system prefix.
    """
    return "\n".join(
        f"- {spec['name']}: {spec['description']}" for spec in TOOL_SPECS
    ) + f"\n- {OUT_OF_SCOPE}: The request is not a medical information request."
