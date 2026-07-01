"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from graphrag_kg import __version__
from graphrag_kg.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=__version__,
    )
