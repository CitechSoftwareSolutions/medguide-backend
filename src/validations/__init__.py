"""Reusable validation rules shared by DTOs and services."""

from .knowledge_validation import clean_tags, require_text

__all__ = ["clean_tags", "require_text"]
