"""Sample data used when the application starts in development."""

from src.dto import CreateKnowledgeEntryRequest
from src.enums import KnowledgeType
from src.repositories import has_entries
from src.services import create_knowledge_entry


def seed_knowledge_entries() -> None:
    """Add a small, idempotent development dataset."""
    if has_entries():
        return

    sample_entries = (
        CreateKnowledgeEntryRequest(
            title="Hypertension",
            summary="Persistently elevated blood pressure that increases cardiovascular risk.",
            knowledge_type=KnowledgeType.CONDITION,
            tags=["cardiovascular", "blood pressure"],
        ),
        CreateKnowledgeEntryRequest(
            title="Paracetamol",
            summary="An analgesic and antipyretic used for mild to moderate pain and fever.",
            knowledge_type=KnowledgeType.MEDICATION,
            tags=["pain relief", "fever"],
        ),
    )
    for entry in sample_entries:
        create_knowledge_entry(entry)
