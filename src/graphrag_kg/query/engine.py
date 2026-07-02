"""Unified QueryEngine for GraphRAG-KG using graphrag 3.x API.

Routes queries to local, global, drift, or basic search with
automatic method selection based on query characteristics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import pandas as pd

from graphrag_kg.core.config import KGConfig
from graphrag_kg.query.cache import QueryCache
from graphrag_kg.query.context import ContextBuilder, QueryResponse

logger = logging.getLogger("graphrag_kg.query.engine")


class QueryEngine:
    """Unified query interface for GraphRAG-KG.

    Routes queries to the appropriate search method and assembles
    grounded responses with citations.

    Usage:
        engine = QueryEngine(config)
        response = engine.ask("What drugs does Hengrui produce?")
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.query_config = config.query
        self.cache = QueryCache(ttl_seconds=config.query.cache_ttl_seconds)
        self.context_builder = ContextBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        method: str = "auto",
        community_level: Optional[int] = None,
    ) -> QueryResponse:
        """Ask a question and get a grounded answer.

        Args:
            question: Natural language question.
            method: 'auto', 'local', 'global', 'drift', 'basic'.
            community_level: Community hierarchy level.

        Returns:
            QueryResponse with answer, citations, and context.
        """
        start_time = time.time()

        # Check cache
        cached = self.cache.get(question, method)
        if cached is not None:
            return cached

        # Route
        effective_method = method
        if method == "auto":
            effective_method = self._auto_route(question)

        logger.info(f"Query: '{question[:80]}...' -> {effective_method}")

        # Execute
        try:
            if effective_method == "local":
                response = asyncio.run(self._local_search(question, community_level))
            elif effective_method == "global":
                response = asyncio.run(self._global_search(question, community_level))
            elif effective_method == "drift":
                response = asyncio.run(self._drift_search(question))
            elif effective_method == "basic":
                response = asyncio.run(self._basic_search(question))
            else:
                response = asyncio.run(self._local_search(question, community_level))
        except Exception as e:
            logger.error(f"Search failed: {e}")
            response = QueryResponse(
                answer=f"Search error: {e}",
                search_method=effective_method,
            )

        response.processing_time_ms = (time.time() - start_time) * 1000
        self.cache.set(question, response, effective_method)
        return response

    # ------------------------------------------------------------------
    # Search Methods (graphrag 3.x async API)
    # ------------------------------------------------------------------

    async def _local_search(
        self, question: str, community_level: Optional[int] = None
    ) -> QueryResponse:
        """Local search: entity expansion + text retrieval."""
        from graphrag.api import local_search

        graphrag_config = self.config.to_graphrag_config()
        entities_df = self._read_parquet("entities")
        communities_df = self._read_parquet("communities")
        reports_df = self._read_parquet("community_reports")
        text_units_df = self._read_parquet("text_units")
        relationships_df = self._read_parquet("relationships")

        # Fall back to basic search if community reports are missing (fast mode)
        if reports_df.empty:
            logger.info("No community_reports found, falling back to basic search")
            return await self._basic_search(question)

        level = community_level or 2

        result, context = await local_search(
            config=graphrag_config,
            entities=entities_df,
            communities=communities_df,
            community_reports=reports_df,
            text_units=text_units_df,
            relationships=relationships_df,
            covariates=None,
            community_level=level,
            response_type=self.query_config.response_type,
            query=question,
        )

        answer = str(result) if isinstance(result, str) else str(result)
        sources = self._extract_sources(context)

        return self.context_builder.build_response(
            answer=answer,
            search_method="local",
            sources=sources,
        )

    async def _global_search(
        self, question: str, community_level: Optional[int] = None
    ) -> QueryResponse:
        """Global search: map-reduce over community reports."""
        from graphrag.api import global_search

        graphrag_config = self.config.to_graphrag_config()
        entities_df = self._read_parquet("entities")
        communities_df = self._read_parquet("communities")
        reports_df = self._read_parquet("community_reports")

        # Fall back to basic if community reports missing
        if reports_df.empty:
            return await self._basic_search(question)

        level = community_level or 2

        result, context = await global_search(
            config=graphrag_config,
            entities=entities_df,
            communities=communities_df,
            community_reports=reports_df,
            community_level=level,
            dynamic_community_selection=False,
            response_type=self.query_config.response_type,
            query=question,
        )

        answer = str(result) if isinstance(result, str) else str(result)
        sources = self._extract_sources(context)

        return self.context_builder.build_response(
            answer=answer,
            search_method="global",
            sources=sources,
        )

    async def _drift_search(self, question: str) -> QueryResponse:
        """DRIFT search: hierarchical community traversal."""
        from graphrag.api import drift_search

        graphrag_config = self.config.to_graphrag_config()
        entities_df = self._read_parquet("entities")
        communities_df = self._read_parquet("communities")
        reports_df = self._read_parquet("community_reports")
        text_units_df = self._read_parquet("text_units")
        relationships_df = self._read_parquet("relationships")

        result, context = await drift_search(
            config=graphrag_config,
            entities=entities_df,
            communities=communities_df,
            community_reports=reports_df,
            text_units=text_units_df,
            relationships=relationships_df,
            community_level=2,
            response_type=self.query_config.response_type,
            query=question,
        )

        answer = str(result) if isinstance(result, str) else str(result)
        sources = self._extract_sources(context)

        return self.context_builder.build_response(
            answer=answer,
            search_method="drift",
            sources=sources,
        )

    async def _basic_search(self, question: str) -> QueryResponse:
        """Basic vector search over text units."""
        from graphrag.api import basic_search

        graphrag_config = self.config.to_graphrag_config()
        text_units_df = self._read_parquet("text_units")

        result, context = await basic_search(
            config=graphrag_config,
            text_units=text_units_df,
            response_type=self.query_config.response_type,
            query=question,
        )

        answer = str(result) if isinstance(result, str) else str(result)
        sources = self._extract_sources(context)

        return self.context_builder.build_response(
            answer=answer,
            search_method="basic",
            sources=sources,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_parquet(self, name: str) -> pd.DataFrame:
        """Read a parquet file, returning empty DataFrame if missing."""
        path = self.config.output_dir / f"{name}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def _extract_sources(self, context: Any) -> list[dict[str, Any]]:
        """Extract source citations from search context data."""
        sources: list[dict[str, Any]] = []
        if context is None:
            return sources

        if isinstance(context, str):
            return [{"text": context[:500]}]

        if isinstance(context, list):
            for item in context:
                if isinstance(item, dict):
                    sources.append(item)
                elif isinstance(item, pd.DataFrame):
                    sources.extend(item.to_dict("records"))
            return sources

        if isinstance(context, dict):
            for key in ["sources", "reports", "entities", "relationships"]:
                if key in context:
                    val = context[key]
                    if isinstance(val, list):
                        sources.extend(val)
                    elif isinstance(val, pd.DataFrame):
                        sources.extend(val.to_dict("records"))

        if isinstance(context, pd.DataFrame):
            sources = context.to_dict("records")

        return sources

    # ------------------------------------------------------------------
    # Auto-Routing
    # ------------------------------------------------------------------

    def _auto_route(self, question: str) -> str:
        """Automatically select the best search method."""
        q_lower = question.lower()

        global_signals = [
            "overview", "summary", "summarize", "trend", "theme",
            "overall", "landscape",
        ]
        if any(s in q_lower for s in global_signals):
            return "global"

        drift_signals = [
            "how does", "chain", "path", "flow", "impact",
            "supply chain", "what would happen", "consequences",
        ]
        if any(s in q_lower for s in drift_signals):
            return "drift"

        basic_signals = ["what is", "who is", "define", "definition", "when was", "where is"]
        if any(s in q_lower for s in basic_signals):
            return "basic"

        return "local"

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Check if the query engine is ready."""
        return (self.config.output_dir / "entities.parquet").exists()

    def get_stats(self) -> dict[str, Any]:
        """Get query engine statistics."""
        return {
            "ready": self.is_ready(),
            "default_method": self.query_config.default_method,
            "cache_stats": self.cache.stats(),
            "output_dir": str(self.config.output_dir),
        }
