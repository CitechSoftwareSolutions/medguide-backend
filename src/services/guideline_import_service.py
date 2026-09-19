"""Turns one structured clinical guideline document into knowledge entries.

A guideline document (see ``ImportGuidelineRequest``) bundles many conditions
plus shared workflow/red-flag/best-practice context in one paste. Retrieval
only searches a knowledge entry's title and summary, so each condition becomes
its own entry with a readable summary flattened from its nested fields, and
the shared context becomes one additional "overview" entry.

``to_knowledge_requests`` is the pure transform and is what the development
seeder uses; ``import_guideline`` adds persistence and indexing on top for the
HTTP endpoint.
"""

from uuid import UUID

from src.dto import CreateKnowledgeEntryRequest
from src.dto.request.guideline_import_request import ConditionEntry, ImportGuidelineRequest
from src.enums import KnowledgeType
from src.models import KnowledgeEntry
from src.repositories import find_by_title, save
from src.services.rag_service import index_document
from src.utils import current_utc_time, new_identifier

_TITLE_MAX_LENGTH = 200


def _join(lines: list[str]) -> str:
    return "; ".join(line.strip() for line in lines if line.strip())


def _condition_summary(condition: ConditionEntry) -> str:
    """Flatten one condition's nested fields into a readable paragraph set."""
    sections = [
        ("Key features", condition.clinical_presentation.key_features),
        ("Common causes", condition.diagnostic_approach.common_causes),
        ("Initial/conservative management", condition.management_plan.initial_conservative),
        ("Specific/advanced management", condition.management_plan.specific_advanced),
        ("Follow-up and monitoring", condition.management_plan.follow_up_monitoring),
    ]
    paragraphs = [f"{label}: {_join(lines)}." for label, lines in sections if lines]
    return "\n\n".join(paragraphs)


def _overview_summary(payload: ImportGuidelineRequest) -> str:
    """Flatten the guideline's shared context into one readable summary."""
    sections: list[str] = []
    if payload.workflow_steps:
        sections.append("Workflow:\n" + "\n".join(payload.workflow_steps))
    if payload.global_red_flags:
        sections.append(f"Red flags requiring urgent attention: {_join(payload.global_red_flags)}.")
    if payload.clinical_best_practices:
        sections.append(f"Best practices: {_join(payload.clinical_best_practices)}.")
    if payload.disclaimer:
        sections.append(payload.disclaimer.strip())
    return "\n\n".join(sections)


def _tags_for(payload: ImportGuidelineRequest) -> list[str]:
    if payload.document_metadata.category:
        return [payload.document_metadata.category.strip().lower()]
    return []


def to_knowledge_requests(
    payload: ImportGuidelineRequest,
) -> list[CreateKnowledgeEntryRequest]:
    """Flatten one guideline document into knowledge-entry requests.

    Produces one request per condition that has any content, plus one overview
    request carrying the document's shared workflow, red flags, best practices
    and disclaimer. Nothing is persisted or indexed here.
    """
    tags = _tags_for(payload)
    requests: list[CreateKnowledgeEntryRequest] = []

    for condition in payload.conditions_registry:
        summary = _condition_summary(condition)
        if not summary:
            continue
        requests.append(
            CreateKnowledgeEntryRequest(
                title=condition.condition_name[:_TITLE_MAX_LENGTH],
                summary=summary,
                knowledge_type=KnowledgeType.CONDITION,
                tags=tags,
            )
        )

    overview_summary = _overview_summary(payload)
    if overview_summary:
        requests.append(
            CreateKnowledgeEntryRequest(
                title=payload.document_metadata.title[:_TITLE_MAX_LENGTH],
                summary=overview_summary,
                knowledge_type=KnowledgeType.PROCEDURE,
                tags=tags,
            )
        )

    return requests


def _get_or_create_id(request: CreateKnowledgeEntryRequest) -> tuple[UUID, bool]:
    """Reuse an existing entry by title, or create one.

    Titles are unique, so re-importing the same guideline never duplicates a
    row here.
    """
    existing = find_by_title(request.title)
    if existing is not None:
        return existing.id, False

    entry = KnowledgeEntry(
        id=new_identifier(),
        title=request.title,
        summary=request.summary,
        knowledge_type=request.knowledge_type,
        tags=request.tags,
        created_at=current_utc_time(),
    )
    saved = save(entry)
    return saved.id, True


def _create_and_index(request: CreateKnowledgeEntryRequest) -> dict[str, object]:
    """Create (or reuse) one entry, indexing it only when it is newly created.

    Re-indexing a reused entry would give it a second, independent
    ``document_id`` in the vector store alongside the one from its first
    import, so every re-paste of the same guideline would silently double its
    chunks there even though the knowledge-entry row was correctly reused.
    Skipping the index step for a reused entry is what actually makes
    re-importing the same document safe.
    """
    entry_id, created = _get_or_create_id(request)
    chunk_count = 0
    if created:
        index_result = index_document(request.title, request.summary, request.tags)
        chunk_count = index_result["chunk_count"]
    return {
        "id": entry_id,
        "title": request.title,
        "chunk_count": chunk_count,
        "created": created,
    }


def import_guideline(payload: ImportGuidelineRequest) -> list[dict[str, object]]:
    """Create a knowledge entry per condition (plus one overview entry), indexing each new one."""
    return [_create_and_index(request) for request in to_knowledge_requests(payload)]
