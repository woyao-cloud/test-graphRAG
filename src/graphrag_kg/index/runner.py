"""Index runner wrapping graphrag 3.x indexing API.

Uses graphrag.api.build_index with GraphRagConfig for building
knowledge graph indexes.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.errors import IndexingError

logger = logging.getLogger("graphrag_kg.index.runner")


class IndexRunner:
    """Wraps graphrag.api.build_index for building knowledge graph indexes.

    Supports standard, fast, and update indexing methods with
    progress callbacks and dry-run validation.
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self._callbacks: list[Callable[[str, dict], None]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        method: str = "standard",
        progress_callback: Optional[Callable[[str, dict], None]] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """Run the graphrag indexing pipeline.

        Args:
            method: 'standard', 'fast', 'standard-update', 'fast-update'.
            progress_callback: Optional callback(step_name, info_dict).
            dry_run: Validate but don't execute.

        Returns:
            Dict with indexing statistics.
        """
        self._validate_input()

        if dry_run:
            return self._dry_run_report()

        if progress_callback:
            self._callbacks.append(progress_callback)

        try:
            self._notify("starting", {"method": method})
            result = asyncio.run(self._run_pipeline(method))
            self._notify("complete", result)
            return result
        except Exception as e:
            self._notify("error", {"error": str(e)})
            raise IndexingError(f"Indexing pipeline failed: {e}") from e

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------

    async def _run_pipeline(self, method: str) -> dict[str, Any]:
        """Execute the graphrag build_index pipeline."""
        from graphrag.api import build_index
        from graphrag.config.enums import IndexingMethod

        # Build graphrag config
        graphrag_config = self.config.to_graphrag_config()

        # Map method to IndexingMethod enum
        method_map = {
            "standard": IndexingMethod.Standard,
            "fast": IndexingMethod.Fast,
            "standard-update": IndexingMethod.Standard,
            "fast-update": IndexingMethod.Fast,
        }
        indexing_method = method_map.get(method, IndexingMethod.Standard)
        is_update = "update" in method

        self._notify("running", {"status": "executing"})

        # Run the pipeline
        await build_index(
            config=graphrag_config,
            method=indexing_method,
            is_update_run=is_update,
            verbose=False,
        )

        # Extract statistics
        return self._extract_statistics()

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _extract_statistics(self) -> dict[str, Any]:
        """Extract indexing statistics from output parquet files."""
        stats: dict[str, Any] = {
            "output_dir": str(self.config.output_dir),
        }

        output_dir = self.config.output_dir

        for name, key in [
            ("entities.parquet", "entity_count"),
            ("relationships.parquet", "relationship_count"),
            ("communities.parquet", "community_count"),
            ("text_units.parquet", "text_unit_count"),
            ("documents.parquet", "document_count"),
        ]:
            path = output_dir / name
            if path.exists():
                try:
                    import pandas as pd
                    df = pd.read_parquet(path)
                    stats[key] = len(df)
                except Exception:
                    stats[key] = 0
            else:
                stats[key] = 0

        return stats

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self) -> None:
        """Validate that input files exist."""
        input_dir = self.config.input_dir
        if not input_dir.exists():
            raise IndexingError(
                f"Input directory does not exist: {input_dir}. "
                f"Run 'graphrag-kg ingest' first."
            )

        txt_files = list(input_dir.glob("*.txt"))
        if not txt_files:
            raise IndexingError(
                f"No .txt files found in {input_dir}. "
                f"Run 'graphrag-kg ingest' first."
            )

        logger.info(f"Found {len(txt_files)} input files in {input_dir}")

    def _dry_run_report(self) -> dict[str, Any]:
        """Generate a dry-run report."""
        input_dir = self.config.input_dir
        txt_files = list(input_dir.glob("*.txt"))
        total_chars = sum(
            len(f.read_text(encoding="utf-8", errors="replace"))
            for f in txt_files
        )

        return {
            "dry_run": True,
            "input_files": len(txt_files),
            "total_characters": total_chars,
            "output_dir": str(self.config.output_dir),
            "chat_model": self.config.chat_model,
            "embedding_model": self.config.embedding_model,
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _notify(self, event: str, data: dict[str, Any]) -> None:
        for cb in self._callbacks:
            try:
                cb(event, data)
            except Exception:
                pass
