"""Tests for Pydantic request DTOs and reusable validation rules."""

import pytest
from pydantic import ValidationError

from src.shared.schemas import IndexDocumentRequest, clean_tags, require_text


def test_index_request_normalizes_text_and_tags() -> None:
    """Incoming fields are trimmed and duplicate tags are removed."""
    payload = IndexDocumentRequest(
        title="  Migraine  ",
        text="  A recurrent headache disorder.  ",
        tags=[" Neurology ", "headache", "neurology"],
    )

    assert payload.title == "Migraine"
    assert payload.text == "A recurrent headache disorder."
    assert payload.tags == ["neurology", "headache"]


def test_index_request_rejects_empty_text_and_unknown_fields() -> None:
    """Pydantic rejects invalid body fields before the service is reached."""
    with pytest.raises(ValidationError):
        IndexDocumentRequest(title="   ", text="A summary.", unexpected="value")


def test_validation_helpers_reject_empty_tags() -> None:
    """Reusable domain rules reject whitespace-only input."""
    with pytest.raises(ValueError):
        require_text("  ")

    with pytest.raises(ValueError):
        clean_tags(["valid", " "])
