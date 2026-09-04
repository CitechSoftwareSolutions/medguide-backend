from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import get_settings
from src.exceptions import register_exception_handlers
from src.middlewares import add_request_timing
from src.routes import knowledge_router
from src.seeders import seed_knowledge_entries


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Seed the sample data once when the application starts."""
    seed_knowledge_entries()
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
