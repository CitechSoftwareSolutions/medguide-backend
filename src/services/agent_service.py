"""Business operations for answering clinical questions."""

from uuid import UUID

from src.agent.agent import AgentAnswer, answer_question
from src.exceptions import KnowledgeBaseEmptyError
from src.services import rag_service


async def answer_clinical_question(
    question: str,
    session_id: UUID | None = None,
) -> AgentAnswer:
    """Answer a question, refusing early when nothing has been indexed.

    The empty-index case is checked here rather than inside the graph: it is a
    configuration problem, not a retrieval outcome, and it deserves a distinct
    status code instead of an abstention that looks like a content gap.
    """
    if rag_service.stats()["chunk_count"] == 0:
        raise KnowledgeBaseEmptyError(
            "No knowledge has been indexed yet, so no question can be answered."
        )

    return await answer_question(question, session_id)
