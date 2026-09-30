"""``POST /api/v1/agent/ask``: the agent's HTTP endpoint."""

from fastapi import APIRouter

from src.agent import agent
from src.agent.schemas import (
    AnswerData,
    AnswerResponse,
    AskQuestionRequest,
    CitationData,
    NiceConflictData,
    NiceValidationData,
)
from src.agent.state import Citation, NiceValidation

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


def _citation(citation: Citation) -> CitationData:
    return CitationData(**citation)


def _nice_validation(validation: NiceValidation) -> NiceValidationData:
    return NiceValidationData(
        status=validation["status"],
        summary=validation["summary"],
        conflicts=[NiceConflictData(**conflict) for conflict in validation["conflicts"]],
        citations=[_citation(citation) for citation in validation["citations"]],
    )


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: AskQuestionRequest) -> AnswerResponse:
    """Answer a clinical question from the knowledge base, checked against NICE.

    Session memory is keyed on ``session_id``: send back the value from a
    previous answer to ask a follow-up in the same conversation.
    """
    result = await agent.answer_clinical_question(payload.question, payload.session_id)
    return AnswerResponse(
        data=AnswerData(
            session_id=result.session_id,
            answer=result.answer,
            citations=[_citation(citation) for citation in result.citations],
            confidence=result.confidence,
            route=result.route,
            nice_validation=_nice_validation(result.nice_validation),
        )
    )
