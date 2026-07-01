"""Global search engine wrapper — map-reduce over community reports.

Best for high-level summarization, theme extraction, and
cross-document understanding.
"""

from __future__ import annotations

import logging
from typing import Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.query.context import ContextBuilder, QueryResponse

logger = logging.getLogger("graphrag_kg.query.global_search")


class GlobalSearchEngine:
    """Global search using map-reduce over community reports.

    Delegates to graphrag's global search engine for:
    - High-level topic summarization
    - Cross-document theme extraction
    - Dataset-wide understanding
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.context_builder = ContextBuilder()

    def search(
        self,
        question: str,
        community_level: Optional[int] = None,
    ) -> QueryResponse:
        """Execute global search over community reports.

        Args:
            question: Natural language question (best for overview/themes).
            community_level: Specific community level to use.

        Returns:
            QueryResponse with answer.
        """
        try:
            from graphrag.query.factory import get_global_search_engine
            from graphrag.query.indexer_adapters import read_indexer_reports

            graphrag_config = self.config.to_graphrag_config()
            reports = read_indexer_reports(
                self.config.output_dir,
                community_level=community_level,
            )

            engine = get_global_search_engine(
                config=graphrag_config,
                reports=reports,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            return self.context_builder.build_response(
                answer=answer,
                search_method="global",
                sources=[],
            )

        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return QueryResponse(
                answer=f"Global search error: {e}",
                search_method="global",
            )
