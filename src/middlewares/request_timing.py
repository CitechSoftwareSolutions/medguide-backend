"""A small response-timing middleware."""

from time import perf_counter

from fastapi import Request, Response


async def add_request_timing(request: Request, call_next) -> Response:
    """Add the time spent processing each request to its response headers."""
    started_at = perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(perf_counter() - started_at) * 1000:.2f}"
    return response
