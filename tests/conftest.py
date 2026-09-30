"""Shared fixtures.

No test in this suite reaches the network or loads the real embedding model.
The model adapter is always stubbed, and embeddings are replaced with a tiny
deterministic vector space so retrieval behaviour can be asserted exactly.
"""

from __future__ import annotations

import math

import pytest

from src.shared.config import get_settings
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_02_nice_guidance import service as tool_02

pytest_plugins = ["tests.graph_stubs"]  # the ``stub`` and ``no_nice_guidance`` fixtures

# A four-axis stand-in for a sentence embedding. Each term owns one axis, so
# cosine similarity between texts is exactly predictable in the assertions.
FAKE_VOCABULARY = ["migraine", "aspirin", "fracture", "asthma"]
FAKE_DIMENSIONS = len(FAKE_VOCABULARY)


def fake_vector(text: str) -> list[float]:
    """Embed text as normalized term counts over the toy vocabulary."""
    lowered = text.lower()
    counts = [float(lowered.count(term)) for term in FAKE_VOCABULARY]
    norm = math.sqrt(sum(value * value for value in counts))
    if norm == 0.0:
        # An out-of-vocabulary text scores zero against everything.
        return [0.0] * FAKE_DIMENSIONS
    return [value / norm for value in counts]


@pytest.fixture
def isolated_vector_store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point both tools' stores at temporary directories with tiny vectors."""
    monkeypatch.setenv("KNOWLEDGE_BASE_VECTOR_STORE_DIR", str(tmp_path / "tool_01_store"))
    monkeypatch.setenv("NICE_VECTOR_STORE_DIR", str(tmp_path / "tool_02_store"))
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(FAKE_DIMENSIONS))
    get_settings.cache_clear()
    tool_01.load()
    tool_02.load()
    yield

    get_settings.cache_clear()
    tool_01.load()
    tool_02.load()


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch):
    """Replace the sentence-transformer with the toy vector space."""
    import numpy as np

    from src.shared import indexing

    monkeypatch.setattr(
        indexing,
        "encode_documents",
        lambda texts: np.array([fake_vector(text) for text in texts], dtype=np.float32),
    )
    monkeypatch.setattr(
        indexing,
        "encode_query",
        lambda text: np.array(fake_vector(text), dtype=np.float32),
    )
