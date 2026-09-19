import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.agent.llm import warm_embedding_model
from src.config import get_settings
from src.exceptions import register_exception_handlers
from src.middlewares import add_request_timing
from src.repositories import vector_repository
from src.routes import agent_router, knowledge_router, rag_router
from src.seeders import seed_knowledge_entries
from src.services import rag_service

logger = logging.getLogger(__name__)


def _prepare_knowledge_index() -> None:
    """Load the embedding model and make the seeded entries searchable."""
    warm_embedding_model()
    vector_repository.load_or_create()
    if rag_service.stats()["chunk_count"] == 0:
        logger.info("Vector store is empty; indexing the seeded knowledge entries")
        rag_service.reindex_from_knowledge_entries()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Seed the sample data and make the knowledge base searchable.

    The embedding model and vector index are prepared here so the first request
    pays no model-load cost.

    A failure here is logged and tolerated rather than aborting startup. Loading
    the embedding model downloads it on first run, so this step depends on the
    network, and the knowledge endpoints need none of it. Taking the whole API
    down because the assistant could not warm up trades a working service for a
    broken one; instead the assistant reports itself unavailable through the
    usual error envelope while everything else keeps serving.
    """
    seed_knowledge_entries()

    try:
        _prepare_knowledge_index()
    except Exception:
        logger.exception(
            "Could not prepare the knowledge index. The API will serve, but "
            "/api/v1/agent/ask will fail until this is resolved."
        )

    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A function-based sample API for medical knowledge entries.",
    lifespan=lifespan,
)
app.middleware("http")(add_request_timing)
register_exception_handlers(app)
app.include_router(knowledge_router)
app.include_router(rag_router)
app.include_router(agent_router)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
    }


@app.get("/", tags=["application"])
def application_info():
    return {
        "message": "Medical Knowledge Base API",
        "documentation": "/docs",
    }
