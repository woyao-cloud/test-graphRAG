"""Basic vector search engine wrapper — pure embedding similarity.

Fastest method — no graph context, just top-k text unit retrieval
by vector similarity.
"""

from __future__ import annotations

import logging

from graphrag_kg.core.config import KGConfig
from graphrag_kg.query.context import ContextBuilder, QueryResponse

logger = logging.getLogger("graphrag_kg.query.basic")


class BasicSearchEngine:
    """Basic vector search over text unit embeddings.

    Delegates to graphrag's basic search engine for:
    - Simple fact retrieval
    - Definition lookup
    - Quick text search without graph context
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.context_builder = ContextBuilder()

    def search(self, question: str, top_k: int = 10) -> QueryResponse:
        """Execute basic vector search.

        Args:
            question: Natural language question (best for simple facts).
            top_k: Number of results to retrieve.

        Returns:
            QueryResponse with answer and text unit sources.
        """
        try:
            from graphrag.query.factory import get_basic_search_engine
            from graphrag.query.indexer_adapters import read_indexer_text_units

            graphrag_config = self.config.to_graphrag_config()
            text_units = read_indexer_text_units(self.config.output_dir)

            engine = get_basic_search_engine(
                config=graphrag_config,
                text_units=text_units,
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
                search_method="basic",
                sources=sources,
            )

        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            return QueryResponse(
                answer=f"Basic search error: {e}",
                search_method="basic",
            )
