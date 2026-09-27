"""Tests for application settings."""

from src.config.settings import Settings


def test_settings_load_with_no_configuration(monkeypatch) -> None:
    """Nothing is required to boot: there is no database to point at."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    Settings(_env_file=None)


def test_agent_defaults_target_the_requested_model() -> None:
    """The agent defaults to Claude Sonnet and the MiniLM embedding model."""
    settings = Settings(_env_file=None)

    assert settings.agent_model == "claude-sonnet-5"
    assert settings.embedding_model_name.endswith("all-MiniLM-L6-v2")
    assert settings.embedding_dimensions == 384


def test_api_key_is_optional_so_settings_load_without_it(monkeypatch) -> None:
    """Configuration must load without a key; the client reports the problem.

    The environment variable is cleared explicitly so the result does not depend
    on whether the developer has a key exported in their shell or in ``.env``.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = Settings(_env_file=None)

    assert settings.anthropic_api_key == ""


def test_retrieval_defaults_are_coherent() -> None:
    """The score floor is a cosine value and the overlap fits inside a chunk."""
    settings = Settings(_env_file=None)

    assert 0.0 <= settings.retrieval_min_score <= 1.0
    assert settings.retrieval_top_k > 0
    assert settings.chunk_overlap < settings.chunk_size
