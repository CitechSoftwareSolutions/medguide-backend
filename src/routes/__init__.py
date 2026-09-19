"""HTTP route declarations."""

from .agent_routes import router as agent_router
from .knowledge_routes import router as knowledge_router
from .rag_routes import router as rag_router

__all__ = ["agent_router", "knowledge_router", "rag_router"]
