"""Document converter for normalizing parsed documents to graphrag input format.

Converts parsed Document objects into plain text files in the graphrag input/
directory, applying optional chunking hints, metadata prepending, and
encoding normalization.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graphrag_kg.ingest.parsers import Document, MarkdownParser


class DocumentConverter:
    """Converts parsed Documents to graphrag-compatible input files.

    GraphRAG expects plain text files in the input/ directory. This converter:
    - Strips markdown formatting (optional)
    - Prepends metadata as structured headers
    - Normalizes encoding and line endings
    - Writes to the configured input directory
    """

    def __init__(
        self,
        input_dir: Path = Path("input"),
        strip_markdown: bool = True,
        prepend_metadata: bool = True,
        encoding: str = "utf-8",
    ):
        self.input_dir = Path(input_dir)
        self.strip_markdown = strip_markdown
        self.prepend_metadata = prepend_metadata
        self.encoding = encoding
        self._markdown_parser = MarkdownParser()

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def convert(self, document: Document) -> Path:
        """Convert a single Document to a graphrag input text file.

        Args:
            document: The parsed Document to convert.

        Returns:
            Path to the output text file.

        Raises:
            OSError: If the output file cannot be written.
        """
        # Prepare text content
        text = document.text

        # Strip markdown if requested
        if self.strip_markdown and document.format in ("md", "html"):
            text = self._markdown_parser.strip_markdown(text)

        # Prepend metadata header
        if self.prepend_metadata:
            text = self._build_metadata_header(document) + text

        # Clean up text
        text = self._clean_text(text)

        # Determine output filename
        output_name = self._make_output_name(document)
        output_path = self.input_dir / output_name

        # Ensure input directory exists
        self.input_dir.mkdir(parents=True, exist_ok=True)

        # Write output
        output_path.write_text(text, encoding=self.encoding)

        return output_path

    def convert_all(
        self,
        documents: list[Document],
        clear_existing: bool = True,
    ) -> list[Path]:
        """Convert multiple documents to graphrag input files.

        Args:
            documents: List of parsed Documents.
            clear_existing: If True, remove existing .txt files in input_dir first.

        Returns:
            List of output file paths.
        """
        # Optionally clear existing input files
        if clear_existing and self.input_dir.exists():
            for existing in self.input_dir.glob("*.txt"):
                try:
                    existing.unlink()
                except OSError:
                    pass

        output_paths: list[Path] = []
        for doc in documents:
            output_path = self.convert(doc)
            output_paths.append(output_path)

        return output_paths

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_output_name(self, document: Document) -> str:
        """Generate a clean output filename from the source document.

        Uses the source filename with format suffix to avoid collisions
        when the same content is ingested in multiple formats.
        """
        source_name = document.source_path.stem
        # Sanitize: replace spaces and special chars with underscores
        safe_name = re.sub(r"[^\w\-.]", "_", source_name)
        # Truncate if too long
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        # Add format suffix for uniqueness
        return f"{safe_name}_{document.format}.txt"

    def _build_metadata_header(self, document: Document) -> str:
        """Build a metadata header to prepend to the document text.

        GraphRAG can use this structured metadata for better chunk attribution.
        """
        header_lines = [
            f"# Document: {document.title}",
            f"# Source: {document.source_path.name}",
            f"# Format: {document.format}",
        ]
        if document.metadata.get("author"):
            header_lines.append(f"# Author: {document.metadata['author']}")
        if document.metadata.get("pages"):
            header_lines.append(f"# Pages: {document.metadata['pages']}")

        header_lines.append(
            f"# Ingested: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )
        header_lines.append("")
        return "\n".join(header_lines) + "\n\n"

    def _clean_text(self, text: str) -> str:
        """Normalize text for graphrag ingestion.

        - Normalize line endings to LF
        - Collapse excessive blank lines
        - Remove control characters (except newlines and tabs)
        - Ensure proper encoding
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse excessive blank lines
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # Remove control characters except \n and \t
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # Ensure trailing newline
        if not text.endswith("\n"):
            text += "\n"

        return text

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def get_conversion_report(
        self,
        documents: list[Document],
        output_paths: list[Path],
    ) -> dict:
        """Generate a conversion summary report."""
        total_input_bytes = sum(d.size_bytes for d in documents)
        total_output_bytes = sum(
            p.stat().st_size for p in output_paths if p.exists()
        )

        return {
            "documents_converted": len(output_paths),
            "input_dir": str(self.input_dir.resolve()),
            "total_input_size_mb": round(total_input_bytes / 1024 / 1024, 2),
            "total_output_size_mb": round(total_output_bytes / 1024 / 1024, 2),
            "strip_markdown": self.strip_markdown,
            "prepend_metadata": self.prepend_metadata,
            "encoding": self.encoding,
        }
