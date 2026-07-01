"""Query endpoints for the REST API."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException

from graphrag_kg.api.models import (
    CitationResponse,
    QueryRequest,
    QueryResponseModel,
    StatsResponse,
)
from graphrag_kg.query.engine import QueryEngine

router = APIRouter(tags=["query"])


def get_query_engine() -> QueryEngine:
    """Dependency: get the query engine instance."""
    from graphrag_kg.core.config import KGConfig
    from graphrag_kg.api.app import get_config

    config = get_config()
    return QueryEngine(config)


@router.post("/query", response_model=QueryResponseModel)
async def query(
    request: QueryRequest,
    engine: QueryEngine = Depends(get_query_engine),
) -> QueryResponseModel:
    """Ask a question and get a grounded answer with citations."""
    if not engine.is_ready():
        raise HTTPException(
            status_code=503,
            detail="No index found. Run indexing first.",
        )

    try:
        response = engine.ask(
            question=request.question,
            method=request.method,
            community_level=request.community_level,
        )

        citations = [
            CitationResponse(
                text_unit_id=c.text_unit_id,
                text=c.text,
                document_name=c.document_name,
                entity_name=c.entity_name,
                relationship=c.relationship,
            )
            for c in response.citations
        ]

        return QueryResponseModel(
            answer=response.answer,
            search_method=response.search_method,
            citations=citations,
            graph_context=response.graph_context.to_context_text() if response.graph_context else None,
            processing_time_ms=response.processing_time_ms,
            llm_usage=response.llm_usage,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats", response_model=StatsResponse)
async def stats(
    engine: QueryEngine = Depends(get_query_engine),
) -> StatsResponse:
    """Get system statistics."""
    try:
        from graphrag_kg.core.config import KGConfig
        from graphrag_kg.storage.parquet_store import ParquetStore
        from graphrag_kg.graph.connection import Neo4jConnection

        config = KGConfig()
        parquet = ParquetStore(config.output_dir)
        index_stats = parquet.get_index_stats()

        neo4j_connected = False
        try:
            conn = Neo4jConnection(config.neo4j)
            conn.connect()
            neo4j_connected = conn.is_connected
            conn.close()
        except Exception:
            pass

        return StatsResponse(
            project_name=config.project_name,
            index_built=parquet.is_indexed(),
            entity_count=index_stats.get("entities", 0),
            relationship_count=index_stats.get("relationships", 0),
            community_count=index_stats.get("communities", 0),
            document_count=index_stats.get("documents", 0),
            text_unit_count=index_stats.get("text_units", 0),
            neo4j_connected=neo4j_connected,
            query_engine_ready=engine.is_ready(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
