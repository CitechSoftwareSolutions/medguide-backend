"""Tests for application settings and database URL conversion."""

from src.config.settings import Settings


def test_converts_standard_postgresql_url_to_psycopg_url() -> None:
    """A Neon-style PostgreSQL URL should select the Psycopg SQLAlchemy driver."""
    settings = Settings(database_url="postgresql://user:password@host.example/database")

    assert settings.sqlalchemy_database_url == (
        "postgresql+psycopg://user:password@host.example/database"
    )


def test_keeps_existing_psycopg_url() -> None:
    """An already-normalized URL should not be changed."""
    url = "postgresql+psycopg://user:password@host.example/database"

    assert Settings(database_url=url).sqlalchemy_database_url == url


def test_agent_defaults_target_the_requested_model() -> None:
    """The agent defaults to Claude Sonnet and the MiniLM embedding model."""
    settings = Settings(database_url="postgresql://user:password@host.example/database")

    assert settings.agent_model == "claude-sonnet-5"
    assert settings.embedding_model_name.endswith("all-MiniLM-L6-v2")
    assert settings.embedding_dimensions == 384


def test_api_key_is_optional_so_settings_load_without_it(monkeypatch) -> None:
    """Configuration must load without a key; the client reports the problem.

    The environment variable is cleared explicitly so the result does not depend
    on whether the developer has a key exported in their shell.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    settings = Settings(database_url="postgresql://user:password@host.example/database")

    assert settings.anthropic_api_key == ""


def test_retrieval_defaults_are_coherent() -> None:
    """The score floor is a cosine value and the overlap fits inside a chunk."""
    settings = Settings(database_url="postgresql://user:password@host.example/database")

    assert 0.0 <= settings.retrieval_min_score <= 1.0
    assert settings.retrieval_top_k > 0
    assert settings.chunk_overlap < settings.chunk_size
