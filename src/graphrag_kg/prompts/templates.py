"""Prompt template registry for GraphRAG-KG.

Manages custom prompt templates for entity extraction, community
summarization, and search methods. Templates are loaded from the
prompts/ directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Default prompt template directory
DEFAULT_PROMPT_DIR = Path(__file__).parent / "defaults"

# Standard prompt file names
PROMPT_FILES = {
    "extract_graph": "extract_graph.txt",
    "summarize_descriptions": "summarize_descriptions.txt",
    "community_report_graph": "community_report_graph.txt",
    "community_report_text": "community_report_text.txt",
    "local_search": "local_search.txt",
    "global_search_map": "global_search_map.txt",
    "global_search_reduce": "global_search_reduce.txt",
    "drift_search": "drift_search.txt",
    "drift_reduce": "drift_reduce.txt",
    "basic_search": "basic_search.txt",
}


class PromptRegistry:
    """Registry for loading and managing prompt templates."""

    def __init__(self, prompts_dir: Optional[Path] = None):
        self.prompts_dir = prompts_dir or DEFAULT_PROMPT_DIR

    def get_prompt(self, name: str) -> str:
        """Load a prompt template by name.

        Args:
            name: Prompt name (e.g., 'extract_graph', 'local_search').

        Returns:
            Prompt template text.

        Raises:
            FileNotFoundError: If prompt file doesn't exist.
        """
        filename = PROMPT_FILES.get(name, f"{name}.txt")
        path = self.prompts_dir / filename

        if not path.exists():
            # Try default directory as fallback
            path = DEFAULT_PROMPT_DIR / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Prompt template not found: {name} ({filename}). "
                f"Check prompts/ directory."
            )

        return path.read_text(encoding="utf-8")

    def list_available(self) -> list[str]:
        """List available prompt names."""
        available = []
        for name, filename in PROMPT_FILES.items():
            if (self.prompts_dir / filename).exists():
                available.append(name)
            elif (DEFAULT_PROMPT_DIR / filename).exists():
                available.append(name)
        return sorted(available)

    def write_defaults(self, target_dir: Path) -> int:
        """Write default prompt templates to a directory.

        Args:
            target_dir: Directory to write prompt files.

        Returns:
            Number of files written.
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for filename in PROMPT_FILES.values():
            source = DEFAULT_PROMPT_DIR / filename
            if source.exists():
                (target_dir / filename).write_text(
                    source.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                count += 1

        return count
