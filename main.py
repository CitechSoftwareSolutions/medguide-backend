import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
    """Seed the sample data and make the knowledge base searchable.

    A failure here is logged and tolerated rather than aborting startup.
    Loading the embedding model downloads it on first run, so this step depends
    on the network, and the knowledge endpoints need none of it. Taking the
    whole API down because the assistant could not warm up trades a working
    service for a broken one; instead the assistant reports itself unavailable
    through the usual error envelope while everything else keeps serving.
    """
    try:
        seed_knowledge_entries()
        warm_embedding_model()
        vector_repository.load_or_create()
        if rag_service.stats()["chunk_count"] == 0:
            logger.info("Vector store is empty; indexing the seeded knowledge entries")
            rag_service.reindex_from_knowledge_entries()
        logger.info("Knowledge index ready")
    except Exception:
        logger.exception(
            "Could not prepare the knowledge index. The API will serve, but "
            "/api/v1/agent/ask will fail until this is resolved."
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bind the port immediately and warm the knowledge index in the background.

    Startup must not block: an ASGI server does not accept connections until
    this function reaches its yield, and a platform that health-checks the port
    will kill a deploy that takes too long to listen. Downloading the embedding
    model and embedding every entry takes minutes on a small instance, so that
    work runs on a worker thread instead of on the startup path.

    Until it finishes, retrieval simply finds an empty index and /agent/ask
    answers with the usual 409 knowledge_base_empty rather than hanging.
    """
    warmup = asyncio.create_task(asyncio.to_thread(_prepare_knowledge_index))

    yield

    warmup.cancel()


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="A function-based sample API for medical knowledge entries.",
    lifespan=lifespan,
)
app.middleware("http")(add_request_timing)

# The browser frontend is served from its own origin in production, so it
# needs permission to call this API. Origins come from configuration rather
# than being hard-coded, so a new deployment URL is one environment variable.
# Credentials stay off: the API uses no cookies, and enabling them would
# forbid the wildcard some hosts rely on for preview deployments.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms"],
)

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
