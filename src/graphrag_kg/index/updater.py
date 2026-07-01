"""Incremental index update logic for GraphRAG-KG.

Supports adding new documents to an existing index without full re-indexing
using graphrag's standard-update and fast-update methods.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graphrag_kg.core.config import KGConfig
from graphrag_kg.index.runner import IndexRunner

logger = logging.getLogger("graphrag_kg.index.updater")


class IndexUpdater:
    """Manages incremental updates to an existing knowledge graph index.

    Tracks which documents have been indexed and routes new documents
    through graphrag's update pipeline.
    """

    STATE_FILE = "index_state.json"

    def __init__(self, config: KGConfig):
        self.config = config
        self.runner = IndexRunner(config)
        self.state_path = config.output_dir / self.STATE_FILE

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        method: str = "standard-update",
        backup: bool = True,
    ) -> dict:
        """Run an incremental index update for new or modified documents.

        Args:
            method: Update method ('standard-update' or 'fast-update').
            backup: Backup existing output before updating.

        Returns:
            Dict with update results and statistics.
        """
        # Check if we have a previous index
        previous_state = self._load_state()
        if not previous_state:
            logger.info("No previous index found, running full index")
            return self.runner.run(method="standard")

        # Detect new documents
        new_docs = self._detect_new_documents(previous_state)
        if not new_docs:
            logger.info("No new documents detected, index is up to date")
            return {
                "updated": False,
                "reason": "no_new_documents",
                "previous_state": previous_state,
            }

        logger.info(f"Detected {len(new_docs)} new documents for update")

        # Backup existing output
        if backup:
            self._backup_output()

        # Run update pipeline
        result = self.runner.run(method=method)

        # Update state
        self._save_state({
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "document_count": previous_state.get("document_count", 0) + len(new_docs),
            "input_files": self._get_current_files(),
            "method": method,
        })

        result["updated"] = True
        result["new_documents"] = len(new_docs)
        return result

    # ------------------------------------------------------------------
    # Document Tracking
    # ------------------------------------------------------------------

    def _detect_new_documents(self, previous_state: dict) -> list[str]:
        """Detect documents added since the last index."""
        previous_files = set(previous_state.get("input_files", []))
        current_files = set(self._get_current_files())
        return sorted(current_files - previous_files)

    def _get_current_files(self) -> list[str]:
        """Get list of current input files with modification times."""
        files = []
        input_dir = self.config.input_dir
        if input_dir.exists():
            for f in sorted(input_dir.glob("*.txt")):
                try:
                    mtime = f.stat().st_mtime
                    files.append(f"{f.name}:{mtime}")
                except OSError:
                    files.append(f.name)
        return files

    # ------------------------------------------------------------------
    # State Management
    # ------------------------------------------------------------------

    def _load_state(self) -> Optional[dict]:
        """Load the previous index state."""
        if not self.state_path.exists():
            return None

        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _save_state(self, state: dict) -> None:
        """Save the current index state."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    # ------------------------------------------------------------------
    # Backup
    # ------------------------------------------------------------------

    def _backup_output(self) -> Optional[Path]:
        """Create a timestamped backup of the output directory."""
        output_dir = self.config.output_dir
        if not output_dir.exists():
            return None

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = output_dir.parent / f"output_backup_{timestamp}"

        try:
            shutil.copytree(output_dir, backup_dir, dirs_exist_ok=True)
            logger.info(f"Backed up output to {backup_dir}")
            return backup_dir
        except OSError as e:
            logger.warning(f"Failed to backup output: {e}")
            return None
