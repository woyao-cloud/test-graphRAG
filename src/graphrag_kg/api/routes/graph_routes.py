"""Neo4j graph management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from graphrag_kg.api.models import GraphStatsResponse, GraphSyncRequest, GraphSyncResponse
from graphrag_kg.core.config import KGConfig

router = APIRouter(tags=["graph"])


@router.post("/graph/sync", response_model=GraphSyncResponse)
async def sync_graph(request: GraphSyncRequest) -> GraphSyncResponse:
    """Sync indexed data from Parquet to Neo4j."""
    try:
        config = KGConfig()
        from graphrag_kg.graph.sync import Neo4jGraphSync

        syncer = Neo4jGraphSync(config)
        results = syncer.sync_all(clear_first=request.clear_first)

        return GraphSyncResponse(
            status="completed",
            entities=results.get("entities", 0),
            relationships=results.get("relationships", 0),
            communities=results.get("communities", 0),
            documents=results.get("documents", 0),
            text_units=results.get("text_units", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats() -> GraphStatsResponse:
    """Get Neo4j graph statistics."""
    try:
        config = KGConfig()
        from graphrag_kg.graph.connection import Neo4jConnection

        conn = Neo4jConnection(config.neo4j)
        try:
            conn.connect()
            health = conn.health_check()

            if not health["connected"]:
                return GraphStatsResponse(connected=False)

            with conn.session() as session:
                from graphrag_kg.graph.queries import CypherQueries
                queries = CypherQueries(session)
                stats = queries.get_graph_stats()

                return GraphStatsResponse(
                    connected=True,
                    total_nodes=stats["total_nodes"],
                    entities=stats["entities"],
                    relationships=stats["relationships"],
                    communities=stats["communities"],
                    neo4j_version=health.get("neo4j_version", ""),
                )
        finally:
            conn.close()

    except Exception:
        return GraphStatsResponse(connected=False)
