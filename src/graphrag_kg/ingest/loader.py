"""Document loader for discovering and parsing documents from source directories.

Walks source directories, matches file patterns, dispatches to parsers,
and returns a collection of parsed Document objects with progress reporting.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Callable, Optional

from graphrag_kg.ingest.parsers import Document, ParserRegistry


class DocumentLoader:
    """Discovers and parses documents from configured source directories.

    Attributes:
        parser_registry: Registry of format-specific parsers.
        max_file_size_mb: Skip files larger than this.
        on_progress: Optional callback for progress reporting.
    """

    def __init__(
        self,
        parser_registry: Optional[ParserRegistry] = None,
        max_file_size_mb: int = 50,
        on_progress: Optional[Callable[[str, int, int], None]] = None,
    ):
        self.parser_registry = parser_registry or ParserRegistry()
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.on_progress = on_progress

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discover(
        self,
        source_directories: list[Path],
        file_patterns: Optional[list[str]] = None,
        recursive: bool = True,
    ) -> list[Path]:
        """Discover document files matching patterns in source directories.

        Args:
            source_directories: Directories to scan.
            file_patterns: Glob patterns (e.g. ["**/*.pdf", "**/*.md"]).
                           Defaults to all supported formats.
            recursive: Whether to scan subdirectories.

        Returns:
            Sorted list of matching file paths (with duplicates removed).
        """
        if file_patterns is None:
            file_patterns = [
                f"**/*{ext}" for ext in self.parser_registry.supported_extensions
            ]

        discovered: set[Path] = set()

        for source_dir in source_directories:
            source_dir = Path(source_dir).resolve()
            if not source_dir.exists():
                continue

            for pattern in file_patterns:
                if recursive:
                    matches = source_dir.glob(pattern)
                else:
                    matches = source_dir.glob(pattern.replace("**/", ""))

                for match in matches:
                    if match.is_file():
                        discovered.add(match.resolve())

        return sorted(discovered)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        source_directories: list[Path],
        file_patterns: Optional[list[str]] = None,
        recursive: bool = True,
        encoding: str = "utf-8",
    ) -> list[Document]:
        """Discover and parse all documents from source directories.

        Args:
            source_directories: Directories to scan.
            file_patterns: Glob patterns for file matching.
            recursive: Scan subdirectories.
            encoding: Default text encoding.

        Returns:
            List of parsed Document objects.
        """
        # Step 1: Discover files
        files = self.discover(source_directories, file_patterns, recursive)

        if not files:
            return []

        # Step 2: Filter by size
        valid_files = self._filter_by_size(files)

        # Step 3: Parse each file
        documents: list[Document] = []
        errors: list[tuple[Path, str]] = []

        total = len(valid_files)
        for i, file_path in enumerate(valid_files):
            if self.on_progress:
                self.on_progress(f"Parsing {file_path.name}", i + 1, total)

            try:
                doc = self.parser_registry.parse(file_path, encoding=encoding)
                if not doc.is_empty:
                    documents.append(doc)
            except Exception as e:
                errors.append((file_path, str(e)))

        return documents

    def load_single(self, path: Path, encoding: str = "utf-8") -> Document:
        """Parse a single file.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If no parser is available for the format.
        """
        path = Path(path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.stat().st_size > self.max_file_size_bytes:
            raise ValueError(
                f"File too large: {path.stat().st_size / 1024 / 1024:.1f}MB "
                f"(max: {self.max_file_size_bytes / 1024 / 1024:.0f}MB)"
            )

        return self.parser_registry.parse(path, encoding=encoding)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_by_size(self, files: list[Path]) -> list[Path]:
        """Filter out files exceeding the size limit."""
        valid = []
        for f in files:
            try:
                size = f.stat().st_size
                if size <= self.max_file_size_bytes:
                    valid.append(f)
            except OSError:
                continue
        return valid

    def get_load_report(
        self,
        documents: list[Document],
        errors: Optional[list[tuple[Path, str]]] = None,
    ) -> dict:
        """Generate a summary report of loaded documents.

        Args:
            documents: List of successfully parsed documents.
            errors: Optional list of (path, error_message) tuples.

        Returns:
            Dict with counts by format, total text length, and error list.
        """
        by_format: dict[str, int] = {}
        total_chars = 0
        empty_docs = 0

        for doc in documents:
            fmt = doc.format
            by_format[fmt] = by_format.get(fmt, 0) + 1
            total_chars += doc.text_length
            if doc.is_empty:
                empty_docs += 1

        return {
            "total_documents": len(documents),
            "total_characters": total_chars,
            "empty_documents": empty_docs,
            "by_format": by_format,
            "errors": [
                {"path": str(p), "error": e}
                for p, e in (errors or [])
            ],
        }
