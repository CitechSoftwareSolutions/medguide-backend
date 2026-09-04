"""Tests for Pydantic request DTOs and reusable validation rules."""

import pytest
from pydantic import ValidationError

from src.dto import CreateKnowledgeEntryRequest
from src.enums import KnowledgeType
from src.validations import clean_tags, require_text


def test_create_request_normalizes_text_and_tags() -> None:
    """Incoming fields are trimmed and duplicate tags are removed."""
    payload = CreateKnowledgeEntryRequest(
        title="  Migraine  ",
        summary="  A recurrent headache disorder.  ",
        knowledge_type=KnowledgeType.CONDITION,
        tags=[" Neurology ", "headache", "neurology"],
    )

    assert payload.title == "Migraine"
    assert payload.summary == "A recurrent headache disorder."
    assert payload.tags == ["neurology", "headache"]


def test_create_request_rejects_empty_text_and_unknown_fields() -> None:
    """Pydantic rejects invalid body fields before the service is reached."""
    with pytest.raises(ValidationError):
        CreateKnowledgeEntryRequest(
            title="   ",
            summary="A summary.",
            knowledge_type=KnowledgeType.CONDITION,
            unexpected="value",
        )


def test_validation_helpers_reject_empty_tags() -> None:
    """Reusable domain rules reject whitespace-only input."""
    with pytest.raises(ValueError):
        require_text("  ")

    with pytest.raises(ValueError):
        clean_tags(["valid", " "])
