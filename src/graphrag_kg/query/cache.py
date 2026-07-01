"""Query result caching with TTL-based expiration.

Caches QueryResponse objects keyed by (question, method) tuples
to avoid redundant LLM calls for repeated queries.
"""

from __future__ import annotations

import hashlib
import threading
import time
from typing import Optional

from graphrag_kg.query.context import QueryResponse


class QueryCache:
    """In-memory cache for query results with TTL expiration.

    Thread-safe. Automatically evicts expired entries on read.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self._cache: dict[str, tuple[float, QueryResponse]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, question: str, method: str = "local") -> Optional[QueryResponse]:
        """Get a cached response if not expired.

        Returns None if not cached or expired.
        """
        if self.ttl <= 0:
            return None

        key = self._make_key(question, method)
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return None

            timestamp, response = entry
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                return None

            return response

    def set(
        self, question: str, response: QueryResponse, method: str = "local"
    ) -> None:
        """Cache a query response."""
        if self.ttl <= 0:
            return

        key = self._make_key(question, method)
        with self._lock:
            self._cache[key] = (time.time(), response)

    def invalidate(self, question: Optional[str] = None) -> int:
        """Invalidate cached entries.

        Args:
            question: If provided, invalidate only entries for this question.
                      If None, invalidate all entries.

        Returns:
            Number of entries invalidated.
        """
        with self._lock:
            if question is None:
                count = len(self._cache)
                self._cache.clear()
                return count

            count = 0
            # Build keys for all methods for this question
            for method in ["local", "global", "drift", "basic", "auto"]:
                key = self._make_key(question, method)
                if key in self._cache:
                    del self._cache[key]
                    count += 1
            return count

    def clear(self) -> int:
        """Clear all cached entries."""
        return self.invalidate()

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            expired = sum(
                1 for ts, _ in self._cache.values()
                if time.time() - ts > self.ttl
            )
            return {
                "total_entries": total,
                "expired_entries": expired,
                "active_entries": total - expired,
                "ttl_seconds": self.ttl,
            }

    def cleanup(self) -> int:
        """Remove all expired entries."""
        with self._lock:
            now = time.time()
            expired_keys = [
                k for k, (ts, _) in self._cache.items()
                if now - ts > self.ttl
            ]
            for k in expired_keys:
                del self._cache[k]
            return len(expired_keys)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_key(self, question: str, method: str) -> str:
        """Create a deterministic cache key."""
        raw = f"{question}|{method}"
        return hashlib.sha256(raw.encode()).hexdigest()
