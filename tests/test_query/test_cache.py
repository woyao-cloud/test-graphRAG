"""Tests for QueryCache."""

from __future__ import annotations

import time

from graphrag_kg.query.cache import QueryCache
from graphrag_kg.query.context import QueryResponse


class TestQueryCache:
    def test_get_miss(self):
        cache = QueryCache(ttl_seconds=3600)
        assert cache.get("test question") is None

    def test_set_and_get(self):
        cache = QueryCache(ttl_seconds=3600)
        response = QueryResponse(answer="cached answer", search_method="local")
        cache.set("test question", response)
        cached = cache.get("test question")
        assert cached is not None
        assert cached.answer == "cached answer"

    def test_different_methods(self):
        """Different methods should have different cache keys."""
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q", QueryResponse(answer="local", search_method="local"), method="local")
        cache.set("q", QueryResponse(answer="global", search_method="global"), method="global")

        assert cache.get("q", "local").answer == "local"
        assert cache.get("q", "global").answer == "global"

    def test_ttl_expiry(self):
        cache = QueryCache(ttl_seconds=0)  # Immediately expired
        cache.set("q", QueryResponse(answer="x"))
        assert cache.get("q") is None

    def test_disabled_cache(self):
        cache = QueryCache(ttl_seconds=0)
        cache.set("q", QueryResponse(answer="x"))
        assert cache.get("q") is None

    def test_invalidate_all(self):
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q1", QueryResponse(answer="a1"))
        cache.set("q2", QueryResponse(answer="a2"))
        count = cache.invalidate()
        assert count == 2
        assert cache.get("q1") is None

    def test_invalidate_specific(self):
        cache = QueryCache(ttl_seconds=3600)
        cache.set("question one", QueryResponse(answer="a1"))
        cache.set("question two", QueryResponse(answer="a2"))
        count = cache.invalidate("question one")
        assert count == 1
        assert cache.get("question one") is None
        assert cache.get("question two") is not None

    def test_stats(self):
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q1", QueryResponse(answer="a"))
        cache.set("q2", QueryResponse(answer="b"))
        stats = cache.stats()
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2

    def test_cleanup(self):
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q1", QueryResponse(answer="a"))
        # Force expiry by manipulating internal timestamp
        cache._cache[cache._make_key("q1", "local")] = (0, QueryResponse(answer="a"))
        count = cache.cleanup()
        assert count == 1

    def test_clear(self):
        cache = QueryCache(ttl_seconds=3600)
        cache.set("q1", QueryResponse(answer="a"))
        count = cache.clear()
        assert count == 1
