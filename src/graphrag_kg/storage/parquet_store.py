"""Parquet file management for graphrag output data.

Provides typed access to graphrag's output parquet files (entities,
relationships, communities, community_reports, text_units, documents)
with pandas DataFrame conversion.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from graphrag_kg.core.errors import StorageError

logger = logging.getLogger("graphrag_kg.storage.parquet")


class ParquetStore:
    """Manages reading/writing of graphrag's parquet output files.

    GraphRAG produces these parquet files in the output directory:
    - entities.parquet
    - relationships.parquet
    - communities.parquet
    - community_reports.parquet
    - text_units.parquet
    - documents.parquet
    - covariates.parquet (optional)
    """

    # Standard graphrag output files
    STANDARD_FILES = [
        "entities",
        "relationships",
        "communities",
        "community_reports",
        "text_units",
        "documents",
    ]

    OPTIONAL_FILES = ["covariates"]

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read(self, name: str) -> pd.DataFrame:
        """Read a parquet file by logical name (e.g., 'entities').

        Args:
            name: Logical name — 'entities', 'relationships', 'communities', etc.

        Returns:
            pandas DataFrame with the parquet data.

        Raises:
            StorageError: If file doesn't exist or can't be read.
        """
        path = self._path_for(name)
        if not path.exists():
            raise StorageError(
                f"Parquet file not found: {path}. "
                f"Run indexing first."
            )
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise StorageError(f"Failed to read {path}: {e}") from e

    def read_entities(self) -> pd.DataFrame:
        """Read entities.parquet."""
        return self.read("entities")

    def read_relationships(self) -> pd.DataFrame:
        """Read relationships.parquet."""
        return self.read("relationships")

    def read_communities(self) -> pd.DataFrame:
        """Read communities.parquet."""
        return self.read("communities")

    def read_community_reports(self) -> pd.DataFrame:
        """Read community_reports.parquet."""
        return self.read("community_reports")

    def read_text_units(self) -> pd.DataFrame:
        """Read text_units.parquet."""
        return self.read("text_units")

    def read_documents(self) -> pd.DataFrame:
        """Read documents.parquet."""
        return self.read("documents")

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write(self, df: pd.DataFrame, name: str) -> Path:
        """Write a DataFrame to a parquet file.

        Args:
            df: DataFrame to write.
            name: Logical name for the file.

        Returns:
            Path to the written file.
        """
        path = self._path_for(name)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Wrote {len(df)} rows to {path}")
        return path

    # ------------------------------------------------------------------
    # Query Helpers
    # ------------------------------------------------------------------

    def get_entity_by_name(self, name: str) -> Optional[dict[str, Any]]:
        """Look up an entity by name."""
        df = self.read_entities()
        # Try common entity name columns
        for col in ["name", "title", "entity"]:
            if col in df.columns:
                matches = df[df[col] == name]
                if len(matches) > 0:
                    return matches.iloc[0].to_dict()
        return None

    def get_entity_by_id(self, entity_id: str) -> Optional[dict[str, Any]]:
        """Look up an entity by ID."""
        df = self.read_entities()
        for col in ["id", "human_readable_id", "short_id"]:
            if col in df.columns:
                matches = df[df[col].astype(str) == str(entity_id)]
                if len(matches) > 0:
                    return matches.iloc[0].to_dict()
        return None

    def get_relationships_for_entity(self, entity_name: str) -> pd.DataFrame:
        """Get all relationships involving an entity."""
        df = self.read_relationships()
        source_col = None
        target_col = None

        for col in df.columns:
            if col in ("source", "source_name"):
                source_col = col
            if col in ("target", "target_name"):
                target_col = col

        if source_col and target_col:
            return df[
                (df[source_col] == entity_name) |
                (df[target_col] == entity_name)
            ]
        return pd.DataFrame()

    def get_community_entities(self, community_id: int) -> list[str]:
        """Get entity names in a community."""
        df = self.read_communities()
        for col in ["entity_ids", "entities", "title"]:
            if col in df.columns:
                matches = df[df["community"] == community_id] if "community" in df.columns else df[df["id"] == community_id]
                if len(matches) > 0:
                    val = matches.iloc[0][col]
                    if isinstance(val, list):
                        return val
                    if isinstance(val, str):
                        return [val]
        return []

    # ------------------------------------------------------------------
    # Index Status
    # ------------------------------------------------------------------

    def is_indexed(self) -> bool:
        """Check if an index exists in the output directory."""
        return all(
            self._path_for(name).exists()
            for name in ["entities", "relationships", "text_units"]
        )

    def get_index_stats(self) -> dict[str, int]:
        """Get row counts for all available parquet files."""
        stats = {}
        for name in self.STANDARD_FILES + self.OPTIONAL_FILES:
            path = self._path_for(name)
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    stats[name] = len(df)
                except Exception:
                    stats[name] = 0
            else:
                stats[name] = 0
        return stats

    def list_available(self) -> list[str]:
        """List available parquet files in the output directory."""
        available = []
        for name in self.STANDARD_FILES + self.OPTIONAL_FILES:
            if self._path_for(name).exists():
                available.append(name)
        return available

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _path_for(self, name: str) -> Path:
        """Get the full path for a named parquet file."""
        return self.output_dir / f"{name}.parquet"
