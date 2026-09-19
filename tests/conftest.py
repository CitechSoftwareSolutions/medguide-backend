"""Shared fixtures for the agent and retrieval tests.

No test in this suite reaches the network. The Claude adapter is always stubbed,
and embeddings are replaced with a tiny deterministic vector space so retrieval
behaviour can be asserted exactly.
"""

from __future__ import annotations

import math

import pytest

from src.config.settings import get_settings
from src.repositories import vector_repository

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
        # An out-of-vocabulary query must still be a unit vector, but one that
        # is orthogonal to every indexed term so it scores zero everywhere.
        return [0.0] * (FAKE_DIMENSIONS - 1) + [0.0]
    return [value / norm for value in counts]


@pytest.fixture
def isolated_vector_store(tmp_path, monkeypatch: pytest.MonkeyPatch):
    """Point the vector store at a temporary directory with tiny vectors."""
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vector_store"))
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", str(FAKE_DIMENSIONS))
    get_settings.cache_clear()

    # Rebuild the module-level index against the new directory and dimensions.
    vector_repository.load_or_create()
    yield

    get_settings.cache_clear()
    vector_repository.load_or_create()


@pytest.fixture
def fake_embeddings(monkeypatch: pytest.MonkeyPatch):
    """Replace the sentence-transformer with the toy vector space."""
    import numpy as np

    from src.services import rag_service

    monkeypatch.setattr(
        rag_service,
        "encode_documents",
        lambda texts: np.array([fake_vector(text) for text in texts], dtype=np.float32),
    )
    monkeypatch.setattr(
        rag_service,
        "encode_query",
        lambda text: np.array(fake_vector(text), dtype=np.float32),
    )
