"""Tests for application settings."""

from src.config.settings import Settings


def test_settings_load_with_no_configuration() -> None:
    """Nothing is required to boot: no database and no API key."""
    Settings(_env_file=None)


def test_agent_defaults_target_the_requested_model() -> None:
    """The agent defaults to local MedGemma and the MiniLM embedding model."""
    settings = Settings(_env_file=None)

    assert settings.ollama_model == "medgemma1.5:4b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.embedding_model_name.endswith("all-MiniLM-L6-v2")
    assert settings.embedding_dimensions == 384



def test_retrieval_defaults_are_coherent() -> None:
    """The score floor is a cosine value and the overlap fits inside a chunk."""
    settings = Settings(_env_file=None)

    assert 0.0 <= settings.retrieval_min_score <= 1.0
    assert settings.retrieval_top_k > 0
    assert settings.chunk_overlap < settings.chunk_size
