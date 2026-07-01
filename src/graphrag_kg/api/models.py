"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================================
# Query
# ============================================================================


class QueryRequest(BaseModel):
    """Request model for POST /query."""

    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    method: str = Field(
        default="auto",
        pattern="^(auto|local|global|drift|basic)$",
        description="Search method",
    )
    community_level: Optional[int] = Field(default=None, ge=0, description="Community hierarchy level")
    include_sources: bool = Field(default=True, description="Include source citations")
    streaming: bool = Field(default=False, description="Stream response tokens")


class CitationResponse(BaseModel):
    """Response model for a source citation."""

    text_unit_id: str = ""
    text: str = ""
    document_name: str = ""
    entity_name: str = ""
    relationship: str = ""


class QueryResponseModel(BaseModel):
    """Response model for POST /query."""

    answer: str
    search_method: str
    citations: list[CitationResponse] = []
    graph_context: Optional[str] = None
    processing_time_ms: float = 0.0
    llm_usage: dict[str, int] = Field(default_factory=dict)


# ============================================================================
# Index
# ============================================================================


class IndexRequest(BaseModel):
    """Request model for POST /index."""

    method: str = Field(
        default="standard",
        pattern="^(standard|fast|standard-update|fast-update)$",
        description="Indexing method",
    )
    clear_input: bool = Field(default=False, description="Clear input/ before indexing")


class IndexResponse(BaseModel):
    """Response model for POST /index."""

    status: str  # "started", "completed", "failed"
    method: str
    message: str = ""
    entity_count: int = 0
    relationship_count: int = 0
    community_count: int = 0


class IndexStatusResponse(BaseModel):
    """Response model for GET /index/status."""

    indexed: bool
    entity_count: int = 0
    relationship_count: int = 0
    community_count: int = 0
    text_unit_count: int = 0
    output_dir: str = ""


# ============================================================================
# Graph
# ============================================================================


class GraphSyncRequest(BaseModel):
    """Request model for POST /graph/sync."""

    clear_first: bool = Field(default=False, description="Clear Neo4j before syncing")


class GraphSyncResponse(BaseModel):
    """Response model for POST /graph/sync."""

    status: str
    entities: int = 0
    relationships: int = 0
    communities: int = 0
    documents: int = 0
    text_units: int = 0


class GraphStatsResponse(BaseModel):
    """Response model for GET /graph/stats."""

    connected: bool
    total_nodes: int = 0
    entities: int = 0
    relationships: int = 0
    communities: int = 0
    neo4j_version: str = ""


# ============================================================================
# Health
# ============================================================================


class HealthResponse(BaseModel):
    """Response model for GET /health."""

    status: str = "ok"
    version: str = ""
    neo4j_connected: bool = False
    index_built: bool = False


# ============================================================================
# Stats
# ============================================================================


class StatsResponse(BaseModel):
    """Response model for GET /stats."""

    project_name: str = ""
    index_built: bool = False
    entity_count: int = 0
    relationship_count: int = 0
    community_count: int = 0
    document_count: int = 0
    text_unit_count: int = 0
    neo4j_connected: bool = False
    query_engine_ready: bool = False
