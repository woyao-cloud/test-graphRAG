"""Unified QueryEngine with auto-routing for GraphRAG-KG.

Provides a single entry point for all four search methods (local, global,
drift, basic) with automatic method selection based on query characteristics.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from graphrag_kg.core.config import KGConfig, QueryConfig
from graphrag_kg.query.cache import QueryCache
from graphrag_kg.query.context import ContextBuilder, GraphContext, QueryResponse

logger = logging.getLogger("graphrag_kg.query.engine")


class QueryEngine:
    """Unified query interface for GraphRAG-KG.

    Routes queries to the appropriate search method (local, global, drift,
    basic) and assembles grounded responses with citations and graph context.

    Usage:
        engine = QueryEngine(config)
        response = engine.ask("What drugs does Hengrui produce?")
        print(response.answer)
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self.query_config = config.query
        self.cache = QueryCache(ttl_seconds=config.query.cache_ttl_seconds)
        self.context_builder = ContextBuilder()

        # Lazy-loaded search engines
        self._local_engine: Any = None
        self._global_engine: Any = None
        self._drift_engine: Any = None
        self._basic_engine: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        method: str = "auto",
        community_level: Optional[int] = None,
        streaming: bool = False,
    ) -> QueryResponse:
        """Ask a question and get a grounded answer.

        Args:
            question: Natural language question.
            method: Search method — 'auto', 'local', 'global', 'drift', 'basic'.
            community_level: Optional community hierarchy level for global search.
            streaming: If True, stream response tokens (not yet implemented).

        Returns:
            QueryResponse with answer, citations, and graph context.
        """
        start_time = time.time()

        # Check cache
        if not streaming:
            cached = self.cache.get(question, method)
            if cached is not None:
                logger.info(f"Cache hit for: {question[:50]}...")
                return cached

        # Route to appropriate method
        effective_method = method
        if method == "auto":
            effective_method = self._auto_route(question)

        logger.info(f"Query: '{question[:80]}...' -> method={effective_method}")

        # Execute search
        if effective_method == "local":
            response = self.ask_local(question)
        elif effective_method == "global":
            response = self.ask_global(question, community_level)
        elif effective_method == "drift":
            response = self.ask_drift(question)
        elif effective_method == "basic":
            response = self.ask_basic(question)
        else:
            response = self.ask_local(question)  # Fallback

        # Record timing
        response.processing_time_ms = (time.time() - start_time) * 1000

        # Cache the result
        if not streaming:
            self.cache.set(question, response, effective_method)

        return response

    def ask_local(self, question: str) -> QueryResponse:
        """Local search: entity expansion + text retrieval.

        Expands from matching entities to neighbors and uses
        community context for grounded answers.
        """
        try:
            from graphrag.query.factory import get_local_search_engine
            from graphrag.query.indexer_adapters import read_indexer_entities, read_indexer_reports

            config = self.config.to_graphrag_config()
            entities = read_indexer_entities(
                self.config.output_dir, community_level=2
            )
            reports = read_indexer_reports(
                self.config.output_dir, community_level=2
            )

            engine = get_local_search_engine(
                config=config,
                reports=reports,
                entities=entities,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            # Extract context data
            context_data = getattr(result, 'context_data', {})
            sources = context_data.get('sources', []) if isinstance(context_data, dict) else []

            return self.context_builder.build_response(
                answer=answer,
                search_method="local",
                sources=sources,
                processing_time_ms=0.0,
            )

        except ImportError as e:
            logger.warning(f"graphrag.query not available: {e}")
            return QueryResponse(
                answer=self._fallback_answer(question, "local"),
                search_method="local",
            )
        except Exception as e:
            logger.error(f"Local search failed: {e}")
            return QueryResponse(
                answer=f"Local search encountered an error: {e}",
                search_method="local",
            )

    def ask_global(
        self, question: str, community_level: Optional[int] = None
    ) -> QueryResponse:
        """Global search: map-reduce over community reports.

        Best for high-level summarization questions about themes,
        trends, and overall content.
        """
        try:
            from graphrag.query.factory import get_global_search_engine
            from graphrag.query.indexer_adapters import read_indexer_reports

            config = self.config.to_graphrag_config()
            reports = read_indexer_reports(
                self.config.output_dir,
                community_level=community_level,
            )

            engine = get_global_search_engine(
                config=config,
                reports=reports,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            return self.context_builder.build_response(
                answer=answer,
                search_method="global",
                sources=[],
                processing_time_ms=0.0,
            )

        except ImportError:
            return QueryResponse(
                answer=self._fallback_answer(question, "global"),
                search_method="global",
            )
        except Exception as e:
            logger.error(f"Global search failed: {e}")
            return QueryResponse(
                answer=f"Global search encountered an error: {e}",
                search_method="global",
            )

    def ask_drift(self, question: str) -> QueryResponse:
        """DRIFT search: hierarchical community traversal.

        Best for complex multi-hop questions requiring deep
        context from multiple community levels.
        """
        try:
            from graphrag.query.factory import get_drift_search_engine
            from graphrag.query.indexer_adapters import (
                read_indexer_entities,
                read_indexer_reports,
                read_indexer_relationships,
            )

            config = self.config.to_graphrag_config()
            entities = read_indexer_entities(self.config.output_dir)
            reports = read_indexer_reports(self.config.output_dir)
            relationships = read_indexer_relationships(self.config.output_dir)

            engine = get_drift_search_engine(
                config=config,
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
                processing_time_ms=0.0,
            )

        except ImportError:
            return QueryResponse(
                answer=self._fallback_answer(question, "drift"),
                search_method="drift",
            )
        except Exception as e:
            logger.error(f"DRIFT search failed: {e}")
            return QueryResponse(
                answer=f"DRIFT search encountered an error: {e}",
                search_method="drift",
            )

    def ask_basic(self, question: str) -> QueryResponse:
        """Basic vector search over text unit embeddings.

        Fastest method — pure vector similarity without graph context.
        Good for simple fact retrieval.
        """
        try:
            from graphrag.query.factory import get_basic_search_engine
            from graphrag.query.indexer_adapters import read_indexer_text_units

            config = self.config.to_graphrag_config()
            text_units = read_indexer_text_units(self.config.output_dir)

            engine = get_basic_search_engine(
                config=config,
                text_units=text_units,
            )

            result = engine.search(question)
            answer = str(result.response) if hasattr(result, 'response') else str(result)

            # Extract text unit sources
            context_data = getattr(result, 'context_data', {})
            sources = context_data.get('sources', []) if isinstance(context_data, dict) else []

            return self.context_builder.build_response(
                answer=answer,
                search_method="basic",
                sources=sources,
                processing_time_ms=0.0,
            )

        except ImportError:
            return QueryResponse(
                answer=self._fallback_answer(question, "basic"),
                search_method="basic",
            )
        except Exception as e:
            logger.error(f"Basic search failed: {e}")
            return QueryResponse(
                answer=f"Basic search encountered an error: {e}",
                search_method="basic",
            )

    # ------------------------------------------------------------------
    # Auto-Routing
    # ------------------------------------------------------------------

    def _auto_route(self, question: str) -> str:
        """Automatically select the best search method for a question.

        Heuristics:
        - Questions with 'overview', 'summary', 'trends' → global
        - Questions with 'how', 'chain', 'path', 'supply chain' → drift
        - Questions with 'what is', 'who is', simple facts → basic
        - Default → local (best general-purpose)
        """
        q_lower = question.lower()

        # Global search signals: high-level summarization
        global_signals = [
            "overview", "summary", "summarize", "trend", "theme",
            "overall", "landscape", "概括", "总结", "总体",
        ]
        if any(s in q_lower for s in global_signals):
            return "global"

        # DRIFT search signals: multi-hop, complex reasoning
        drift_signals = [
            "how does", "chain", "path", "flow", "impact",
            "supply chain", "what would happen", "consequences",
            "供应链", "路径", "流程", "影响",
        ]
        if any(s in q_lower for s in drift_signals):
            return "drift"

        # Basic search signals: simple fact lookup
        basic_signals = [
            "what is", "who is", "define", "definition",
            "when was", "where is",
        ]
        if any(s in q_lower for s in basic_signals):
            return "basic"

        # Default: local search
        return "local"

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_answer(self, question: str, method: str) -> str:
        """Generate a fallback answer when graphrag query is unavailable.

        This is used when the graphrag library's query modules aren't
        installed or when an index hasn't been built yet.
        """
        return (
            f"[{method.upper()} SEARCH NOT AVAILABLE]\n\n"
            f"To answer: \"{question}\"\n\n"
            f"Please ensure:\n"
            f"1. GraphRAG index has been built: graphrag-kg index\n"
            f"2. Required dependencies are installed: pip install graphrag\n"
            f"3. Output directory contains parquet files: {self.config.output_dir}"
        )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Check if the query engine is ready (index exists)."""
        entities_path = self.config.output_dir / "entities.parquet"
        return entities_path.exists()

    def get_stats(self) -> dict[str, Any]:
        """Get query engine statistics."""
        return {
            "ready": self.is_ready(),
            "default_method": self.query_config.default_method,
            "cache_stats": self.cache.stats(),
            "output_dir": str(self.config.output_dir),
        }
