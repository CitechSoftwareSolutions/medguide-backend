"""Configuration values read when the application starts."""

from functools import lru_cache
from os import getenv

from pydantic import BaseModel


class Settings(BaseModel):
    """Runtime configuration for this small API."""

    app_name: str = "Medical Knowledge Base API"
    environment: str = "development"


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings object for the running process."""
    return Settings(
        app_name=getenv("APP_NAME", "Medical Knowledge Base API"),
        environment=getenv("APP_ENVIRONMENT", "development"),
    )
