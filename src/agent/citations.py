"""Numbered passages and the ``[n]`` citation markers that refer to them.

The model sees passages as ``[1] Title\\ntext``, ``[2] ...`` and cites them by
number. These helpers render passages that way and resolve the numbers back to
chunks. Used by the answer path (Tool 01 passages) and by Tool 02's check.
"""

from __future__ import annotations

import re

from src.agent.state import Citation, RetrievedChunk

CITATION_MARKER = re.compile(r"\[(\d+)\]")


def format_passages(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as numbered passages the model can cite by number."""
    return "\n\n".join(
        f"[{number}] {chunk['title']}\n{chunk['text']}"
        for number, chunk in enumerate(chunks, start=1)
    )


def resolve_markers(markers: list[int], chunks: list[RetrievedChunk]) -> list[Citation]:
    """Turn passage numbers into citations, dropping repeats and bad numbers."""
    seen: list[int] = []
    for marker in markers:
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


def citations_in(text: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Resolve the bracketed markers a text actually used.

    Only markers present in the text become citations, so a response never
    claims provenance the text did not rely on. Out-of-range markers are dropped
    rather than trusted.
    """
    return resolve_markers([int(raw) for raw in CITATION_MARKER.findall(text)], chunks)


def strip_markers(text: str) -> str:
    """Remove ``[n]`` markers and the space before each one."""
    return re.sub(r"\s*\[\d+\]", "", text).strip()
