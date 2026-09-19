"""Functions that shape agent results into API response DTOs."""

from uuid import UUID

from src.dto import AnswerData, AnswerResponse, CitationData
from src.services import agent_service


async def ask(question: str, session_id: UUID | None = None) -> AnswerResponse:
    """Answer one clinical question."""
    result = await agent_service.answer_clinical_question(question, session_id)
    return AnswerResponse(
        data=AnswerData(
            session_id=result.session_id,
            answer=result.answer,
            citations=[
                CitationData(
                    marker=citation["marker"],
                    chunk_id=citation["chunk_id"],
                    document_id=citation["document_id"],
                    title=citation["title"],
                    score=citation["score"],
                )
                for citation in result.citations
            ],
            confidence=result.confidence,
            route=result.route,
        )
    )
