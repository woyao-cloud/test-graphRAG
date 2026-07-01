"""Index runner wrapping Microsoft GraphRAG's indexing pipeline.

Provides a clean interface for running graphrag.index pipelines with
progress monitoring, error handling, and configuration management.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.errors import IndexingError

logger = logging.getLogger("graphrag_kg.index.runner")


class IndexRunner:
    """Wraps graphrag.index API for building knowledge graph indexes.

    Handles:
    - Pipeline configuration from KGConfig
    - Standard and fast indexing methods
    - Incremental updates
    - Progress callbacks
    - Error recovery
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
            method: Indexing method — 'standard', 'fast', 'standard-update', 'fast-update'.
            progress_callback: Optional callback(step_name, info_dict) for progress.
            dry_run: If True, validate config and input but don't run pipeline.

        Returns:
            Dict with indexing results (entity_count, relationship_count, etc.).

        Raises:
            IndexingError: If pipeline fails or configuration is invalid.
        """
        self._validate_input()

        if dry_run:
            return self._dry_run_report()

        if progress_callback:
            self._callbacks.append(progress_callback)

        try:
            result = self._run_pipeline(method)
            self._notify("complete", result)
            return result
        except Exception as e:
            self._notify("error", {"error": str(e)})
            raise IndexingError(f"Indexing pipeline failed: {e}") from e

    def run_fast(self, **kwargs: Any) -> dict[str, Any]:
        """Run fast indexing (fewer gleanings, smaller chunks)."""
        return self.run(method="fast", **kwargs)

    def run_update(self, **kwargs: Any) -> dict[str, Any]:
        """Run incremental update indexing."""
        return self.run(method="standard-update", **kwargs)

    # ------------------------------------------------------------------
    # Pipeline Execution
    # ------------------------------------------------------------------

    def _run_pipeline(self, method: str) -> dict[str, Any]:
        """Execute the graphrag pipeline with the given method."""
        # Build graphrag configuration
        graphrag_config = self.config.to_graphrag_config()

        # Select workflow list based on method
        workflows = self._get_workflows_for_method(method)

        self._notify("starting", {"method": method, "workflows": len(workflows)})

        # Run the graphrag pipeline
        try:
            from graphrag.index import run_pipeline_with_config

            self._notify("running", {"status": "executing"})

            # Run pipeline
            results = run_pipeline_with_config(
                config=graphrag_config,
                workflows=workflows,
            )

            # Extract statistics from output
            stats = self._extract_statistics()
            self._notify("statistics", stats)

            return stats

        except ImportError:
            raise IndexingError(
                "graphrag.index module not available. "
                "Install graphrag: pip install graphrag>=3.0.0"
            )

    def _get_workflows_for_method(self, method: str) -> list[Any]:
        """Get the workflow list for the given indexing method.

        Maps method names to graphrag workflow configurations.
        """
        from graphrag.index.config import PipelineWorkflowReference

        # Standard workflow pipeline
        standard_workflows = [
            PipelineWorkflowReference(name="load_input_documents"),
            PipelineWorkflowReference(name="create_base_text_units"),
            PipelineWorkflowReference(name="extract_graph"),
            PipelineWorkflowReference(name="finalize_graph"),
            PipelineWorkflowReference(name="summarize_descriptions"),
            PipelineWorkflowReference(name="create_communities"),
            PipelineWorkflowReference(name="create_community_reports"),
            PipelineWorkflowReference(name="generate_text_embeddings"),
            PipelineWorkflowReference(name="create_final_documents"),
            PipelineWorkflowReference(name="create_final_text_units"),
        ]

        # Fast workflow pipeline (skip community reports)
        fast_workflows = [
            PipelineWorkflowReference(name="load_input_documents"),
            PipelineWorkflowReference(name="create_base_text_units"),
            PipelineWorkflowReference(name="extract_graph"),
            PipelineWorkflowReference(name="finalize_graph"),
            PipelineWorkflowReference(name="summarize_descriptions"),
            PipelineWorkflowReference(name="create_communities"),
            PipelineWorkflowReference(name="generate_text_embeddings"),
            PipelineWorkflowReference(name="create_final_documents"),
            PipelineWorkflowReference(name="create_final_text_units"),
        ]

        # Update workflows (add incremental steps)
        update_workflows = standard_workflows.copy()
        update_workflows.insert(0, PipelineWorkflowReference(name="create_base_document_nodes"))
        update_workflows.insert(1, PipelineWorkflowReference(name="update_text_units"))

        if "fast" in method:
            return fast_workflows
        elif "update" in method:
            return update_workflows
        return standard_workflows

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _extract_statistics(self) -> dict[str, Any]:
        """Extract indexing statistics from output files."""
        stats: dict[str, Any] = {
            "method": "standard",
            "output_dir": str(self.config.output_dir),
        }

        output_dir = self.config.output_dir

        # Read entity count
        entities_path = output_dir / "entities.parquet"
        if entities_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(entities_path)
                stats["entity_count"] = len(df)
            except Exception:
                stats["entity_count"] = 0
        else:
            stats["entity_count"] = 0

        # Read relationship count
        rels_path = output_dir / "relationships.parquet"
        if rels_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(rels_path)
                stats["relationship_count"] = len(df)
            except Exception:
                stats["relationship_count"] = 0
        else:
            stats["relationship_count"] = 0

        # Read community count
        comms_path = output_dir / "communities.parquet"
        if comms_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(comms_path)
                stats["community_count"] = len(df)
            except Exception:
                stats["community_count"] = 0
        else:
            stats["community_count"] = 0

        # Read text unit count
        tu_path = output_dir / "text_units.parquet"
        if tu_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(tu_path)
                stats["text_unit_count"] = len(df)
            except Exception:
                stats["text_unit_count"] = 0
        else:
            stats["text_unit_count"] = 0

        return stats

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input(self) -> None:
        """Validate that input files exist and config is correct."""
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
        """Generate a dry-run report without executing the pipeline."""
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
        """Notify all registered callbacks."""
        for cb in self._callbacks:
            try:
                cb(event, data)
            except Exception:
                pass
