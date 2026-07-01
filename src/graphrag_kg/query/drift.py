"""DRIFT search engine wrapper — hierarchical community traversal.

Best for complex multi-hop questions requiring deep context
from multiple community hierarchy levels.
"""

from __future__ import annotations

import logging

from graphrag_kg.core.config import KGConfig
from graphrag_kg.query.context import ContextBuilder, QueryResponse

logger = logging.getLogger("graphrag_kg.query.drift")


class DriftSearchEngine:
    """DRIFT search using hierarchical community traversal.

    Delegates to graphrag's DRIFT search engine for:
    - Multi-hop reasoning questions
    - Impact/flow analysis
    - Supply chain tracing
    - Hierarchical context exploration
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.context_builder = ContextBuilder()

    def search(self, question: str) -> QueryResponse:
        """Execute DRIFT search with hierarchical community traversal.

        Args:
            question: Natural language question (best for complex multi-hop).

        Returns:
            QueryResponse with answer.
        """
        try:
            from graphrag.query.factory import get_drift_search_engine
            from graphrag.query.indexer_adapters import (
                read_indexer_entities,
                read_indexer_reports,
                read_indexer_relationships,
            )

            graphrag_config = self.config.to_graphrag_config()
            entities = read_indexer_entities(self.config.output_dir)
            reports = read_indexer_reports(self.config.output_dir)
            relationships = read_indexer_relationships(self.config.output_dir)

            engine = get_drift_search_engine(
                config=graphrag_config,
                entities=entities,
                reports=reports,
                relationships=relationships,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            return self.context_builder.build_response(
                answer=answer,
                search_method="drift",
                sources=[],
            )

        except Exception as e:
            logger.error(f"DRIFT search failed: {e}")
            return QueryResponse(
                answer=f"DRIFT search error: {e}",
                search_method="drift",
            )
