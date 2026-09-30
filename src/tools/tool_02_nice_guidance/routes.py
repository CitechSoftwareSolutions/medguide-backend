"""Tool 02's document endpoints, under ``/api/v1/nice``."""

from src.shared.document_routes import build_document_router
from src.tools.tool_02_nice_guidance import service

router = build_document_router(
    prefix="/api/v1/nice", tag="tool 02: NICE guidance", service=service
)
