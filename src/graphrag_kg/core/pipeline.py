"""Full pipeline orchestrator for end-to-end GraphRAG workflows.

Orchestrates the complete lifecycle: ingest → index → graph sync → query,
with progress reporting, error recovery, and configurable steps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.errors import GraphRAGKGError

logger = logging.getLogger("graphrag_kg.core.pipeline")


class Pipeline:
    """Orchestrates the full GraphRAG-KG pipeline.

    Usage:
        pipeline = Pipeline(config)
        pipeline.run_full()  # ingest → index → sync
        pipeline.run_quick()  # ingest → index (skip Neo4j)
    """

    def __init__(self, config: KGConfig):
        self.config = config
        self._on_step: Optional[Callable[[str, str, dict], None]] = None

    # ------------------------------------------------------------------
    # Full Pipeline
    # ------------------------------------------------------------------

    def run_full(
        self,
        source_dirs: Optional[list[Path]] = None,
        index_method: str = "standard",
        sync_to_neo4j: bool = True,
        clear_neo4j: bool = False,
    ) -> dict[str, Any]:
        """Run the complete pipeline: ingest → index → sync.

        Args:
            source_dirs: Source directories for documents.
            index_method: Indexing method.
            sync_to_neo4j: Whether to sync to Neo4j after indexing.
            clear_neo4j: Clear Neo4j before syncing.

        Returns:
            Dict with results from each step.
        """
        results: dict[str, Any] = {}

        # Step 1: Ingest
        self._notify("ingest", "started", {})
        ingest_result = self._run_ingest(source_dirs)
        results["ingest"] = ingest_result
        self._notify("ingest", "completed", ingest_result)

        # Step 2: Index
        self._notify("index", "started", {"method": index_method})
        index_result = self._run_index(index_method)
        results["index"] = index_result
        self._notify("index", "completed", index_result)

        # Step 3: Sync to Neo4j
        if sync_to_neo4j:
            self._notify("sync", "started", {})
            sync_result = self._run_sync(clear_neo4j)
            results["sync"] = sync_result
            self._notify("sync", "completed", sync_result)

        return results

    def run_quick(
        self,
        source_dirs: Optional[list[Path]] = None,
    ) -> dict[str, Any]:
        """Run a quick pipeline: ingest → index (skip Neo4j)."""
        return self.run_full(
            source_dirs=source_dirs,
            index_method="fast",
            sync_to_neo4j=False,
        )

    def run_update(
        self,
        source_dirs: Optional[list[Path]] = None,
    ) -> dict[str, Any]:
        """Run incremental update: ingest new docs → update index → sync."""
        return self.run_full(
            source_dirs=source_dirs,
            index_method="standard-update",
            sync_to_neo4j=True,
        )

    # ------------------------------------------------------------------
    # Individual Steps
    # ------------------------------------------------------------------

    def _run_ingest(self, source_dirs: Optional[list[Path]] = None) -> dict[str, Any]:
        """Run document ingestion."""
        from graphrag_kg.ingest.loader import DocumentLoader
        from graphrag_kg.ingest.converter import DocumentConverter

        if source_dirs is None:
            source_dirs = [
                self.config.root_dir / d
                for d in self.config.ingestion.source_directories
            ]

        loader = DocumentLoader(max_file_size_mb=self.config.ingestion.max_file_size_mb)
        documents = loader.load(
            source_directories=source_dirs,
            file_patterns=self.config.ingestion.file_patterns,
            recursive=self.config.ingestion.recursive,
            encoding=self.config.ingestion.encoding,
        )

        converter = DocumentConverter(
            input_dir=self.config.input_dir,
            strip_markdown=self.config.ingestion.clean_html,
            prepend_metadata=self.config.ingestion.extract_metadata,
            encoding=self.config.ingestion.encoding,
        )

        output_paths = converter.convert_all(documents)
        report = loader.get_load_report(documents)

        return {
            "documents_parsed": len(documents),
            "input_files_written": len(output_paths),
            "total_characters": report.get("total_characters", 0),
            "by_format": report.get("by_format", {}),
        }

    def _run_index(self, method: str) -> dict[str, Any]:
        """Run knowledge graph indexing."""
        from graphrag_kg.index.runner import IndexRunner

        runner = IndexRunner(self.config)
        return runner.run(method=method)

    def _run_sync(self, clear_first: bool) -> dict[str, Any]:
        """Run Neo4j sync."""
        from graphrag_kg.graph.sync import Neo4jGraphSync

        syncer = Neo4jGraphSync(self.config)
        return syncer.sync_all(clear_first=clear_first)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_step(self, callback: Callable[[str, str, dict], None]) -> None:
        """Register a step callback: callback(step_name, status, data)."""
        self._on_step = callback

    def _notify(self, step: str, status: str, data: dict) -> None:
        """Notify the step callback."""
        if self._on_step:
            try:
                self._on_step(step, status, data)
            except Exception:
                pass
