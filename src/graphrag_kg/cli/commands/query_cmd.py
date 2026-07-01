"""CLI commands for querying the knowledge graph."""

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
)
from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.query.engine import QueryEngine

query_app = typer.Typer(help="Knowledge graph query commands")


@query_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    method: str = typer.Option(
        "auto", "--method", "-m",
        help="Search method: auto, local, global, drift, basic",
    ),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
    community_level: Optional[int] = typer.Option(
        None, "--community-level", "-l",
        help="Community hierarchy level for global search",
    ),
    no_sources: bool = typer.Option(
        False, "--no-sources",
        help="Hide source citations",
    ),
    raw: bool = typer.Option(
        False, "--raw",
        help="Output raw JSON response",
    ),
) -> None:
    """Ask a question and get a grounded answer with citations."""
    # Load config
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("GraphRAG Q&A")
    print_info(f"Question: {question}")
    print_info(f"Method: {method}")

    # Create engine
    engine = QueryEngine(config)

    if not engine.is_ready():
        print_error("No index found. Run 'graphrag-kg index' first.")
        raise typer.Exit(code=1)

    # Execute query
    try:
        response = engine.ask(
            question=question,
            method=method,
            community_level=community_level,
        )

        if raw:
            import json
            console.print_json(json.dumps(response.to_dict(), ensure_ascii=False, indent=2))
        else:
            # Print answer
            console.print()
            console.print(f"[bold]Answer:[/bold]")
            console.print(response.answer)
            console.print()

            # Print metadata
            print_info(f"Method: {response.search_method}")
            print_info(f"Time: {response.processing_time_ms:.0f}ms")

            # Print citations
            if not no_sources and response.citations:
                print_header("Sources")
                for i, citation in enumerate(response.citations[:5], 1):
                    console.print(f"[bold]{i}.[/bold] {citation.to_markdown()}")
                    console.print()

            # Print graph context if available
            if response.graph_context:
                ctx = response.graph_context
                if ctx.related_entities:
                    print_info(f"Related entities: {len(ctx.related_entities)}")
                if ctx.communities:
                    print_info(f"Communities: {len(ctx.communities)}")

    except Exception as e:
        print_error(f"Query failed: {e}")
        raise typer.Exit(code=1)


@query_app.command("status")
def query_status(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
) -> None:
    """Show query engine status."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("Query Engine Status")
    engine = QueryEngine(config)
    stats = engine.get_stats()

    console.print(f"  Ready: [bold green]{stats['ready']}[/bold green]" if stats['ready']
                  else f"  Ready: [bold red]False[/bold red]")
    console.print(f"  Default method: {stats['default_method']}")
    console.print(f"  Output dir: {stats['output_dir']}")

    cache_stats = stats.get("cache_stats", {})
    if cache_stats:
        console.print(f"  Cache entries: {cache_stats.get('active_entries', 0)}")
