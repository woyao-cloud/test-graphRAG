"""CLI commands for document ingestion.

Usage:
    graphrag-kg ingest [--source DIR] [--patterns PATTERNS]
                       [--encoding ENC] [--no-clear] [--dry-run]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from graphrag_kg.cli.utils import (
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_table,
    print_warning,
)
from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.ingest.loader import DocumentLoader
from graphrag_kg.ingest.converter import DocumentConverter

ingest_app = typer.Typer(help="Document ingestion commands")


@ingest_app.command("run")
def ingest(
    source: Optional[list[Path]] = typer.Option(
        None, "--source", "-s",
        help="Source directories to scan (overrides config)",
    ),
    patterns: Optional[str] = typer.Option(
        None, "--patterns", "-p",
        help="Comma-separated glob patterns (e.g. '**/*.pdf,**/*.md')",
    ),
    encoding: str = typer.Option(
        "utf-8", "--encoding", "-e",
        help="Text encoding for reading files",
    ),
    no_clear: bool = typer.Option(
        False, "--no-clear",
        help="Don't clear existing files in input/ before ingesting",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Discover files but don't parse or convert",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
    max_file_size: int = typer.Option(
        50, "--max-size", "-m",
        help="Maximum file size in MB",
    ),
) -> None:
    """Ingest documents into the project for indexing.

    Discovers documents in configured source directories, parses them
    into plain text, and writes them to the input/ directory for
    graphrag indexing.
    """
    # Load config
    loader_cfg = ConfigLoader(config_path)
    config = loader_cfg.load(config_path) if config_path else KGConfig()

    print_header("Document Ingestion")

    # Determine source directories
    if source:
        source_dirs = [Path(s).resolve() for s in source]
    else:
        source_dirs = [config.root_dir / d for d in config.ingestion.source_directories]

    # Determine file patterns
    if patterns:
        file_patterns = [p.strip() for p in patterns.split(",")]
    else:
        file_patterns = config.ingestion.file_patterns

    print_info(f"Source directories: {[str(d) for d in source_dirs]}")
    print_info(f"File patterns: {file_patterns}")
    print_info(f"Encoding: {encoding}")
    print_info(f"Max file size: {max_file_size}MB")

    # Create loader
    doc_loader = DocumentLoader(max_file_size_mb=max_file_size)

    # Discover files
    files = doc_loader.discover(
        source_directories=source_dirs,
        file_patterns=file_patterns,
        recursive=config.ingestion.recursive,
    )

    if not files:
        print_warning("No documents found matching the specified patterns.")
        raise typer.Exit(code=0)

    print_info(f"Discovered {len(files)} files")

    if dry_run:
        print_header("Files to Ingest (dry run)")
        by_ext: dict[str, int] = {}
        for f in files:
            ext = f.suffix.lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1

        rows = [[ext, str(count)] for ext, count in sorted(by_ext.items())]
        print_table("Files by Extension", ["Extension", "Count"], rows)
        return

    # Parse documents
    documents = doc_loader.load(
        source_directories=source_dirs,
        file_patterns=file_patterns,
        recursive=config.ingestion.recursive,
        encoding=encoding,
    )

    if not documents:
        print_warning("No documents could be parsed successfully.")
        raise typer.Exit(code=0)

    print_success(f"Parsed {len(documents)} documents")

    # Convert to graphrag input format
    input_dir = config.input_dir
    converter = DocumentConverter(
        input_dir=input_dir,
        strip_markdown=config.ingestion.clean_html,
        prepend_metadata=config.ingestion.extract_metadata,
        encoding=config.ingestion.encoding,
    )

    output_paths = converter.convert_all(
        documents,
        clear_existing=not no_clear,
    )

    # Print results
    print_header("Ingestion Results")

    load_report = doc_loader.get_load_report(documents)
    conv_report = converter.get_conversion_report(documents, output_paths)

    # Format summary
    by_format = load_report.get("by_format", {})
    if by_format:
        rows = [[fmt, str(count)] for fmt, count in sorted(by_format.items())]
        print_table("Documents by Format", ["Format", "Count"], rows)

    print_success(f"Input files written to: {input_dir.resolve()}")
    print_info(f"Documents: {conv_report['documents_converted']}")
    print_info(f"Total size: {conv_report['total_output_size_mb']}MB")

    if load_report.get("empty_documents", 0) > 0:
        print_warning(
            f"{load_report['empty_documents']} documents had no extractable text"
        )

    if load_report.get("errors"):
        print_warning(f"{len(load_report['errors'])} files had parse errors")


@ingest_app.command("discover")
def discover(
    source: Optional[list[Path]] = typer.Option(
        None, "--source", "-s",
        help="Source directories to scan",
    ),
    patterns: Optional[str] = typer.Option(
        None, "--patterns", "-p",
        help="Comma-separated glob patterns",
    ),
) -> None:
    """Discover documents without parsing them."""
    config = KGConfig()

    if source:
        source_dirs = [Path(s).resolve() for s in source]
    else:
        source_dirs = [config.root_dir / d for d in config.ingestion.source_directories]

    if patterns:
        file_patterns = [p.strip() for p in patterns.split(",")]
    else:
        file_patterns = config.ingestion.file_patterns

    doc_loader = DocumentLoader()
    files = doc_loader.discover(source_dirs, file_patterns)

    print_header(f"Discovered Files ({len(files)})")

    by_ext: dict[str, list[str]] = {}
    for f in files:
        ext = f.suffix.lower()
        by_ext.setdefault(ext, []).append(f.name)

    for ext in sorted(by_ext):
        console.print(f"\n[bold]{ext}[/bold] ({len(by_ext[ext])} files):")
        for name in by_ext[ext][:10]:
            console.print(f"  - {name}")
        if len(by_ext[ext]) > 10:
            console.print(f"  ... and {len(by_ext[ext]) - 10} more")
