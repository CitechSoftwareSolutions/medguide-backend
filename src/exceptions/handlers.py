"""Global FastAPI exception handlers."""

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.exceptions.errors import ApplicationError

logger = logging.getLogger(__name__)


def _error_response(
    *, status_code: int, error: str, message: str, details: Any | None = None
) -> JSONResponse:
    """Create the one error format used by this API."""
    content: dict[str, Any] = {
        "error": error,
        "message": message,
    }
    if details is not None:
        content["details"] = details
    return JSONResponse(status_code=status_code, content=content)


async def handle_application_error(_: Request, exc: ApplicationError) -> JSONResponse:
    """Turn known domain errors into predictable client responses."""
    return _error_response(
        status_code=exc.status_code,
        error=exc.error_code,
        message=exc.message,
    )


async def handle_request_validation_error(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return malformed request data in the application error format."""
    return _error_response(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        error="validation_error",
        message="The request contains invalid data.",
        details=jsonable_encoder(exc.errors()),
    )


async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    """Log unexpected failures without exposing internal details."""
    logger.exception("Unhandled application error", exc_info=exc)
    return _error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        error="internal_server_error",
        message="An unexpected error occurred.",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Attach every global handler in one place during application setup."""
    app.add_exception_handler(ApplicationError, handle_application_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(Exception, handle_unexpected_error)
