"""Domain-specific validation and normalization rules."""


def require_text(value: str) -> str:
    """Return trimmed text, rejecting empty or whitespace-only values."""
    cleaned_value = value.strip()
    if not cleaned_value:
        raise ValueError("This field must contain text.")
    return cleaned_value


def clean_tags(tags: list[str]) -> list[str]:
    """Trim tags, remove duplicates, and reject empty tags."""
    cleaned_tags = [require_text(tag).lower() for tag in tags]
    return list(dict.fromkeys(cleaned_tags))
