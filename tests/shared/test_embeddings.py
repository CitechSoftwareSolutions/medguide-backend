"""The embedding model reports load failures as an unavailable assistant."""

import sys

import pytest

from src.shared import embeddings
from src.shared.errors import AgentUnavailableError


@pytest.fixture
def unloadable_model(monkeypatch: pytest.MonkeyPatch):
    """Make ``import sentence_transformers`` fail, as a blocked DLL does."""
    embeddings.get_embedding_model.cache_clear()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    yield
    embeddings.get_embedding_model.cache_clear()


@pytest.mark.usefixtures("unloadable_model")
def test_model_that_cannot_load_becomes_agent_unavailable():
    with pytest.raises(AgentUnavailableError, match="search model"):
        embeddings.encode_query("asthma")


@pytest.mark.usefixtures("unloadable_model")
def test_load_failure_is_not_cached():
    """A fixed environment recovers on the next request without a restart."""
    for _ in range(2):
        with pytest.raises(AgentUnavailableError):
            embeddings.get_embedding_model()
    assert embeddings.get_embedding_model.cache_info().currsize == 0
