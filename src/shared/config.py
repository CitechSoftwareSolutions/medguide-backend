"""Configuration values read when the application starts."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_TOOLS_DIR = Path(__file__).resolve().parents[1] / "tools"


class Settings(BaseSettings):
    """Runtime configuration, overridable through environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Medical Knowledge Base API"
    environment: str = "development"

    # The agent's language model: MedGemma served by a local Ollama.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "medgemma1.5:4b"
    ollama_num_ctx: int = 8192
    ollama_timeout_seconds: float = 300.0
    ollama_keep_alive: str = "30m"
    # Includes MedGemma's thinking block, which precedes the answer.
    agent_max_answer_tokens: int = 2048
    agent_recursion_limit: int = 40

    # Embeddings and chunking, shared by both tools.
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    chunk_size: int = 500
    chunk_overlap: int = 80
    max_upload_bytes: int = 20 * 1024 * 1024

    # Tool 01: the knowledge base answers are written from.
    knowledge_base_documents_dir: Path = _TOOLS_DIR / "tool_01_knowledge_base" / "data"
    knowledge_base_vector_store_dir: Path = Path("var/tool_01_knowledge_base")
    knowledge_base_top_k: int = 8
    knowledge_base_min_score: float = 0.35

    # Tool 02: NICE guidance answers are checked against, never written from.
    nice_documents_dir: Path = _TOOLS_DIR / "tool_02_nice_guidance" / "data"
    nice_vector_store_dir: Path = Path("var/tool_02_nice_guidance")
    nice_top_k: int = 6
    nice_min_score: float = 0.35

    session_max_turns: int = 20
    session_ttl_seconds: int = 3600

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    cors_origin_regex: str = ""

    @property
    def allowed_origins(self) -> list[str]:
        """The browser origins allowed to call this API.

        Comma-separated so the deployed frontend's URL can be added through
        one environment variable without a code change.
        """
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object for the running process."""
    return Settings()
