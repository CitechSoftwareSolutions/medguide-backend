"""Tool 01's document endpoints, under ``/api/v1/rag``."""

from src.shared.document_routes import build_document_router
from src.tools.tool_01_knowledge_base import service

router = build_document_router(
    prefix="/api/v1/rag", tag="tool 01: knowledge base", service=service
)
