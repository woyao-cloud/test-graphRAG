"""Local search engine wrapper with Neo4j graph enrichment.

Extends graphrag's local search with Neo4j-powered entity expansion,
neighborhood traversal, and community context for richer answers.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.graph.connection import Neo4jConnection
from graphrag_kg.graph.queries import CypherQueries
from graphrag_kg.graph.traversal import GraphTraversal
from graphrag_kg.query.context import ContextBuilder, GraphContext, QueryResponse

logger = logging.getLogger("graphrag_kg.query.local")


class LocalSearchEngine:
    """Local search with optional Neo4j graph enrichment.

    Wraps graphrag's local search engine and optionally enriches results
    with Neo4j graph traversal data for deeper entity context.

    Falls back gracefully if Neo4j is not available.
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.context_builder = ContextBuilder()
        self.neo4j: Optional[Neo4jConnection] = None
        self._init_neo4j()

    def _init_neo4j(self) -> None:
        """Initialize Neo4j connection if available."""
        try:
            self.neo4j = Neo4jConnection(self.config.neo4j)
            self.neo4j.connect()
            logger.info("Neo4j enrichment enabled for local search")
        except Exception:
            logger.info("Neo4j not available, local search will use parquet only")
            self.neo4j = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        question: str,
        top_k_entities: int = 10,
        enrich_with_graph: bool = True,
    ) -> QueryResponse:
        """Execute local search with optional Neo4j enrichment.

        Args:
            question: Natural language question.
            top_k_entities: Number of entities to retrieve.
            enrich_with_graph: Whether to add Neo4j graph context.

        Returns:
            QueryResponse with answer and context.
        """
        # Step 1: Run graphrag local search
        response = self._run_graphrag_local_search(question, top_k_entities)

        # Step 2: Enrich with Neo4j graph data if available
        if enrich_with_graph and self.neo4j and self.neo4j.is_connected:
            graph_context = self._enrich_from_neo4j(response)
            response.graph_context = graph_context

        return response

    def _run_graphrag_local_search(
        self, question: str, top_k_entities: int
    ) -> QueryResponse:
        """Run the native graphrag local search."""
        try:
            from graphrag.query.factory import get_local_search_engine
            from graphrag.query.indexer_adapters import (
                read_indexer_entities,
                read_indexer_reports,
            )

            graphrag_config = self.config.to_graphrag_config()
            entities = read_indexer_entities(
                self.config.output_dir, community_level=2
            )
            reports = read_indexer_reports(
                self.config.output_dir, community_level=2
            )

            engine = get_local_search_engine(
                config=graphrag_config,
                reports=reports,
                entities=entities,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            context_data = getattr(result, 'context_data', {})
            sources = (
                context_data.get('sources', [])
                if isinstance(context_data, dict)
                else []
            )

            return self.context_builder.build_response(
                answer=answer,
                search_method="local",
                sources=sources,
            )

        except Exception as e:
            logger.error(f"graphrag local search failed: {e}")
            return QueryResponse(
                answer=f"Local search error: {e}",
                search_method="local",
            )

    # ------------------------------------------------------------------
    # Neo4j Enrichment
    # ------------------------------------------------------------------

    def _enrich_from_neo4j(self, response: QueryResponse) -> GraphContext:
        """Enrich a QueryResponse with Neo4j graph traversal data.

        Extracts entity names from citations and performs ego network
        expansion to add graph context.
        """
        if not self.neo4j or not self.neo4j.is_connected:
            return GraphContext()

        try:
            with self.neo4j.session() as session:
                queries = CypherQueries(session)
                traversal = GraphTraversal(session)

                # Extract entity names from citations
                entity_names = list(set(
                    c.entity_name for c in response.citations
                    if c.entity_name
                ))

                if not entity_names:
                    return GraphContext()

                # Get ego network for the first matched entity
                main_entity_name = entity_names[0]
                main_entity = queries.get_entity(main_entity_name)

                # Get related entities
                ego = traversal.ego_network(main_entity_name, radius=1)
                related = ego.get("nodes", [])

                # Get community context
                community_ctx = traversal.community_neighborhood(main_entity_name)
                communities = []
                if community_ctx.get("community"):
                    communities.append(community_ctx["community"])
                communities.extend(community_ctx.get("ancestors", []))
                communities.extend(community_ctx.get("descendants", []))

                return GraphContext(
                    source_entity=main_entity,
                    related_entities=related,
                    communities=communities,
                )

        except Exception as e:
            logger.warning(f"Neo4j enrichment failed: {e}")
            return GraphContext()

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.neo4j:
            self.neo4j.close()
