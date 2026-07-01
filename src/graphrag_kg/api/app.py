"""FastAPI application for GraphRAG-KG REST API.

Usage:
    uvicorn graphrag_kg.api.app:app --host 0.0.0.0 --port 8000
    graphrag-kg serve
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from graphrag_kg import __version__
from graphrag_kg.core.config import KGConfig

# Global config reference (set by serve command)
_config: Optional[KGConfig] = None


def get_config() -> KGConfig:
    """Get the current configuration."""
    global _config
    if _config is None:
        _config = KGConfig()
    return _config


def set_config(config: KGConfig) -> None:
    """Set the global configuration."""
    global _config
    _config = config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup and shutdown events."""
    import logging
    logger = logging.getLogger("graphrag_kg.api")
    logger.info(f"GraphRAG-KG API v{__version__} starting...")
    yield
    logger.info("GraphRAG-KG API shutting down...")


app = FastAPI(
    title="GraphRAG-KG API",
    description="GraphRAG Knowledge Graph Q&A System with Neo4j + LanceDB",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Setup middleware
from graphrag_kg.api.middleware import setup_middleware
setup_middleware(app)

# Register routes
from graphrag_kg.api.routes.health import router as health_router
from graphrag_kg.api.routes.query import router as query_router
from graphrag_kg.api.routes.index import router as index_router
from graphrag_kg.api.routes.graph_routes import router as graph_router

app.include_router(health_router)
app.include_router(query_router)
app.include_router(index_router)
app.include_router(graph_router)
