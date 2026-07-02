"""Tests for QueryEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from graphrag_kg.core.config import KGConfig
from graphrag_kg.query.engine import QueryEngine
from graphrag_kg.query.context import QueryResponse


class TestQueryEngine:
    """Tests for QueryEngine (unit tests, no index required)."""

    @pytest.fixture
    def engine(self):
        config = KGConfig()
        return QueryEngine(config)

    def test_engine_creation(self, engine):
        assert engine is not None
        assert engine.query_config.default_method == "local"

    def test_is_ready_no_index(self, engine):
        assert engine.is_ready() is False

    def test_auto_route_global(self, engine):
        """Questions about summaries/trends should route to global."""
        assert engine._auto_route("Give me an overview of the main themes") == "global"
        assert engine._auto_route("Summarize the trends in this dataset") == "global"

    def test_auto_route_drift(self, engine):
        """Questions about chains/paths should route to drift."""
        assert engine._auto_route("How does the supply chain flow?") == "drift"
        assert engine._auto_route("What would happen if the chain broke?") == "drift"

    def test_auto_route_basic(self, engine):
        """Simple fact questions should route to basic."""
        assert engine._auto_route("What is GraphRAG?") == "basic"
        assert engine._auto_route("Who is the CEO?") == "basic"

    def test_auto_route_local_default(self, engine):
        """Unclassified questions should default to local."""
        assert engine._auto_route("Tell me about the company's products") == "local"

    def test_ask_without_index(self, engine):
        """ask() should return error response when no index exists."""
        response = engine.ask("test question", method="local")
        assert response is not None
        assert response.search_method == "local"
        # Should contain error info since no index
        assert "error" in response.answer.lower() or "search error" in response.answer.lower()

    def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert "ready" in stats
        assert "default_method" in stats
        assert "cache_stats" in stats
        assert stats["ready"] is False


class TestQueryEngineWithCache:
    """Tests for QueryEngine cache integration."""

    @pytest.fixture
    def engine(self):
        config = KGConfig()
        config.query.cache_ttl_seconds = 3600
        return QueryEngine(config)

    def test_cache_hit(self, engine):
        """Cache should return previously stored responses."""
        from graphrag_kg.query.context import QueryResponse

        # Pre-populate cache
        engine.cache.set("test q", QueryResponse(answer="cached"), "local")

        # ask() with method that doesn't require index
        # Cache should be checked before actual search
        result = engine.cache.get("test q", "local")
        assert result is not None
        assert result.answer == "cached"


class TestAutoRouting:
    """Comprehensive auto-routing tests."""

    @pytest.fixture
    def engine(self):
        return QueryEngine(KGConfig())

    @pytest.mark.parametrize("question,expected", [
        ("What is the meaning of life?", "basic"),
        ("Who is the CEO of Acme Corp?", "basic"),
        ("When was the company founded?", "basic"),
        ("Give me an overview of the dataset", "global"),
        ("Summarize the main themes", "global"),
        ("What are the key trends?", "global"),
        ("How does the supply chain work?", "drift"),
        ("What would happen if supply is disrupted?", "drift"),
        ("Describe the impact of the merger", "drift"),
        ("What products does the company make?", "local"),
        ("Tell me about the partnership", "local"),
        ("Explain the relationship between A and B", "local"),
    ])
    def test_routing(self, engine, question, expected):
        assert engine._auto_route(question) == expected
