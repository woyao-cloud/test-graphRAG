"""CLI commands for knowledge graph indexing."""

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
    print_warning,
)
from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.index.runner import IndexRunner
from graphrag_kg.index.monitor import PipelineMonitor

index_app = typer.Typer(help="Knowledge graph indexing commands")


@index_app.command("run")
def index(
    method: str = typer.Option(
        "standard", "--method", "-m",
        help="Indexing method: standard, fast, standard-update, fast-update",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Validate config but don't run the pipeline",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v",
        help="Verbose output",
    ),
) -> None:
    """Build the knowledge graph index from ingested documents."""
    # Load config
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header(f"GraphRAG Indexing: {method}")

    # Validate method
    valid_methods = ["standard", "fast", "standard-update", "fast-update"]
    if method not in valid_methods:
        print_error(f"Invalid method '{method}'. Choose from: {valid_methods}")
        raise typer.Exit(code=1)

    # Create monitor
    monitor = PipelineMonitor(verbose=verbose)

    # Create runner
    runner = IndexRunner(config)

    if dry_run:
        print_info("Dry run mode — validating configuration...")
        report = runner.run(method=method, dry_run=True)
        print_header("Dry Run Report")
        for key, value in report.items():
            console.print(f"  [bold]{key}[/bold]: {value}")
        return

    # Run indexing
    print_info(f"Input directory: {config.input_dir}")
    print_info(f"Output directory: {config.output_dir}")
    print_info(f"Chat model: {config.chat_model}")
    print_info(f"Embedding model: {config.embedding_model}")

    try:
        result = runner.run(
            method=method,
            progress_callback=monitor.on_progress,
        )

        print_header("Indexing Results")
        console.print(f"  Entities: [bold green]{result.get('entity_count', 0)}[/bold green]")
        console.print(f"  Relationships: [bold green]{result.get('relationship_count', 0)}[/bold green]")
        console.print(f"  Communities: [bold green]{result.get('community_count', 0)}[/bold green]")
        console.print(f"  Text Units: [bold green]{result.get('text_unit_count', 0)}[/bold green]")
        console.print()
        print_info(f"Output: {config.output_dir}")
        print_success("Indexing complete!")

    except Exception as e:
        print_error(f"Indexing failed: {e}")
        raise typer.Exit(code=1)
