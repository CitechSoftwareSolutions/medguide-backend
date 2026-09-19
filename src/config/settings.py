"""Configuration values read when the application starts."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for this small API."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Medical Knowledge Base API"
    environment: str = "development"
    database_url: str

    anthropic_api_key: str = ""
    agent_model: str = "claude-sonnet-5"
    agent_max_answer_tokens: int = 8000
    agent_recursion_limit: int = 40

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    vector_store_dir: Path = Path("var/vector_store")
    chunk_size: int = 500
    chunk_overlap: int = 80
    retrieval_top_k: int = 8
    retrieval_min_score: float = 0.35

    session_max_turns: int = 20
    session_ttl_seconds: int = 3600

    @property
    def sqlalchemy_database_url(self) -> str:
        """Translate a standard PostgreSQL URL to SQLAlchemy's Psycopg driver URL."""
        if self.database_url.startswith("postgresql+psycopg://"):
            return self.database_url
        if self.database_url.startswith("postgresql://"):
            return self.database_url.replace(
                "postgresql://", "postgresql+psycopg://", 1
            )
        if self.database_url.startswith("postgres://"):
            return self.database_url.replace("postgres://", "postgresql+psycopg://", 1)
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object for the running process."""
    return Settings()
