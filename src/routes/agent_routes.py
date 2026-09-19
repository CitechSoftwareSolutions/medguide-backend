"""HTTP endpoints for the clinical question-answering agent."""

from fastapi import APIRouter

from src.controllers import ask
from src.dto import AnswerResponse, AskQuestionRequest

router = APIRouter(prefix="/api/v1/agent", tags=["agent"])


@router.post("/ask", response_model=AnswerResponse)
async def ask_question(payload: AskQuestionRequest) -> AnswerResponse:
    """Answer a clinical question from the indexed knowledge base.

    Session memory is keyed on ``session_id``: send back the value from a
    previous answer to ask a follow-up in the same conversation.
    """
    return await ask(payload.question, payload.session_id)
