"""Index management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from graphrag_kg.api.models import IndexRequest, IndexResponse, IndexStatusResponse
from graphrag_kg.core.config import KGConfig
from graphrag_kg.storage.parquet_store import ParquetStore

router = APIRouter(tags=["index"])


@router.post("/index", response_model=IndexResponse)
async def run_index(request: IndexRequest) -> IndexResponse:
    """Trigger a knowledge graph indexing job."""
    try:
        config = KGConfig()
        from graphrag_kg.index.runner import IndexRunner

        runner = IndexRunner(config)
        result = runner.run(method=request.method)

        return IndexResponse(
            status="completed",
            method=request.method,
            message="Indexing completed successfully",
            entity_count=result.get("entity_count", 0),
            relationship_count=result.get("relationship_count", 0),
            community_count=result.get("community_count", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/index/status", response_model=IndexStatusResponse)
async def index_status() -> IndexStatusResponse:
    """Get indexing status."""
    config = KGConfig()
    parquet = ParquetStore(config.output_dir)
    stats = parquet.get_index_stats()

    return IndexStatusResponse(
        indexed=parquet.is_indexed(),
        entity_count=stats.get("entities", 0),
        relationship_count=stats.get("relationships", 0),
        community_count=stats.get("communities", 0),
        text_unit_count=stats.get("text_units", 0),
        output_dir=str(config.output_dir),
    )
