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
