"""LanceDB vector store wrapper for graphrag embeddings.

Provides typed access to LanceDB vector tables for text unit,
entity, and community embeddings with search capabilities.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("graphrag_kg.storage.vectors")


class VectorStore:
    """Wrapper around LanceDB for graphrag embedding storage.

    GraphRAG stores embeddings in LanceDB tables under output/lancedb/.
    This provides a clean interface for vector search and retrieval.
    """

    def __init__(self, db_uri: Path):
        self.db_uri = Path(db_uri)
        self._db: Any = None

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def db(self) -> Any:
        """Get or create LanceDB connection (lazy init)."""
        if self._db is None:
            try:
                import lancedb
                self._db = lancedb.connect(str(self.db_uri))
            except ImportError:
                raise ImportError(
                    "LanceDB is required for vector store access. "
                    "Install: pip install lancedb"
                )
        return self._db

    def is_available(self) -> bool:
        """Check if the LanceDB database exists."""
        return self.db_uri.exists()

    # ------------------------------------------------------------------
    # Table Access
    # ------------------------------------------------------------------

    def list_tables(self) -> list[str]:
        """List available LanceDB tables."""
        if not self.is_available():
            return []
        return self.db.table_names()

    def get_table(self, name: str) -> Optional[Any]:
        """Get a LanceDB table by name."""
        if not self.is_available():
            return None
        try:
            return self.db.open_table(name)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        table_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Search a LanceDB table by vector similarity.

        Args:
            table_name: Name of the table to search (e.g., 'text_embeddings').
            query_vector: Query embedding vector.
            top_k: Number of results to return.
            filter_expr: Optional LanceDB filter expression.

        Returns:
            List of result dicts with 'id', 'text', 'distance', and metadata.
        """
        table = self.get_table(table_name)
        if table is None:
            logger.warning(f"Table '{table_name}' not found in {self.db_uri}")
            return []

        try:
            query = table.search(query_vector).limit(top_k)
            if filter_expr:
                query = query.where(filter_expr)
            results = query.to_list()
            return results
        except Exception as e:
            logger.error(f"Vector search failed on {table_name}: {e}")
            return []

    def search_text_embeddings(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search text unit embeddings."""
        for name in ["text_embeddings", "text_units", "description_embedding"]:
            results = self.search(name, query_vector, top_k)
            if results:
                return results
        return []

    def search_entity_embeddings(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search entity description embeddings."""
        for name in ["entity_embeddings", "entities", "entity.description_embedding"]:
            results = self.search(name, query_vector, top_k)
            if results:
                return results
        return []

    def search_community_embeddings(
        self,
        query_vector: list[float],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search community report embeddings."""
        for name in ["community_embeddings", "communities", "community_reports"]:
            results = self.search(name, query_vector, top_k)
            if results:
                return results
        return []

    # ------------------------------------------------------------------
    # Data Access
    # ------------------------------------------------------------------

    def get_by_id(self, table_name: str, record_id: str) -> Optional[dict[str, Any]]:
        """Get a single record by ID."""
        table = self.get_table(table_name)
        if table is None:
            return None

        try:
            # LanceDB filter by id
            results = table.search().where(f"id = '{record_id}'").limit(1).to_list()
            if results:
                return results[0]
        except Exception:
            pass

        return None

    def to_dataframe(self, table_name: str) -> Optional[pd.DataFrame]:
        """Convert a LanceDB table to a pandas DataFrame."""
        table = self.get_table(table_name)
        if table is None:
            return None
        try:
            return table.to_pandas()
        except Exception as e:
            logger.error(f"Failed to convert {table_name}: {e}")
            return None

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get vector store statistics."""
        stats: dict[str, Any] = {
            "available": self.is_available(),
            "db_uri": str(self.db_uri),
            "tables": {},
        }

        for table_name in self.list_tables():
            try:
                table = self.get_table(table_name)
                if table:
                    stats["tables"][table_name] = {
                        "rows": table.count_rows(),
                        "schema": str(table.schema),
                    }
            except Exception:
                stats["tables"][table_name] = {"rows": "unknown"}

        return stats
