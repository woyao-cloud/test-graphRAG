"""FastAPI middleware for logging, CORS, and error handling."""

from __future__ import annotations

import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

logger = logging.getLogger("graphrag_kg.api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Logs request method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = (time.time() - start) * 1000
        logger.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code} ({duration:.0f}ms)"
        )
        return response


def setup_middleware(app):
    """Add standard middleware to the FastAPI app."""
    # CORS - allow all origins for development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging
    app.add_middleware(LoggingMiddleware)
