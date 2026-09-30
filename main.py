"""The FastAPI application: startup, middleware and routes.

Layout: ``src/agent`` orchestrates, ``src/tools`` holds Tool 01 (knowledge base)
and Tool 02 (NICE guidance), and ``src/shared`` holds what both use.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.ollama_client import warm_chat_model
from src.agent.routes import router as agent_router
from src.shared.config import get_settings
from src.shared.embeddings import warm_embedding_model
from src.shared.error_handlers import register_exception_handlers
from src.shared.request_timing import add_request_timing
from src.tools.tool_01_knowledge_base import service as tool_01
from src.tools.tool_01_knowledge_base.routes import router as tool_01_router
from src.tools.tool_02_nice_guidance import service as tool_02
from src.tools.tool_02_nice_guidance.routes import router as tool_02_router

logger = logging.getLogger(__name__)


def _prepare_knowledge_index() -> None:
    """Index both tools' PDFs and make them searchable.

    A failure here is logged and tolerated rather than aborting startup.
    Loading the embedding model downloads it on first run, so this step depends
    on the network, and the knowledge endpoints need none of it. Taking the
    whole API down because the assistant could not warm up trades a working
    service for a broken one; instead the assistant reports itself unavailable
    through the usual error envelope while everything else keeps serving.
    """
    try:
        warm_embedding_model()
        tool_01.load()
        tool_01.sync()
    except Exception:
        logger.exception(
            "Could not prepare the knowledge index. The API will serve, but "
            "/api/v1/agent/ask will fail until this is resolved."
        )
        return

    # Tool 02 only checks answers, so a failure here is logged and the assistant
    # keeps answering; each answer then reports the check unavailable or not
    # covered.
    try:
        tool_02.load()
        tool_02.sync()
    except Exception:
        logger.exception("Could not prepare the NICE index; answers will not be validated")

    try:
        warm_chat_model()
    except Exception:
        logger.exception("Could not preload the chat model; the first question will be slow")
    logger.info("Knowledge index ready")


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
    description="Question answering over clinical guideline PDFs indexed in FAISS.",
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
app.include_router(agent_router)
app.include_router(tool_01_router)
app.include_router(tool_02_router)


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
