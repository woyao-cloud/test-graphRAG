"""CLI commands for configuration management."""

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
from graphrag_kg.core.config import KGConfig, PROFILES
from graphrag_kg.core.config_loader import ConfigLoader

config_app = typer.Typer(help="Configuration management commands")


@config_app.command("show")
def show(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
) -> None:
    """Display the current effective configuration."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("Current Configuration")

    console.print(f"[bold]Project:[/bold] {config.project_name}")
    console.print(f"[bold]Description:[/bold] {config.description or '(none)'}")
    console.print()

    console.print("[bold]LLM:[/bold]")
    console.print(f"  Chat model: {config.chat_model} ({config.chat_model_provider})")
    console.print(f"  Embedding: {config.embedding_model} ({config.embedding_model_provider})")
    console.print()

    console.print("[bold]Neo4j:[/bold]")
    console.print(f"  URI: {config.neo4j.uri}")
    console.print(f"  Database: {config.neo4j.database}")
    console.print(f"  Pool size: {config.neo4j.max_connection_pool_size}")
    console.print()

    console.print("[bold]Ingestion:[/bold]")
    console.print(f"  Sources: {config.ingestion.source_directories}")
    console.print(f"  Patterns: {config.ingestion.file_patterns}")
    console.print()

    console.print("[bold]Pipeline:[/bold]")
    console.print(f"  Auto-index: {config.pipeline.auto_index_on_ingest}")
    console.print(f"  Max workers: {config.pipeline.max_workers}")
    console.print()

    console.print("[bold]Query:[/bold]")
    console.print(f"  Default method: {config.query.default_method}")
    console.print(f"  Max context tokens: {config.query.max_context_tokens}")
    console.print(f"  Sources: {config.query.include_sources}")
    console.print()


@config_app.command("validate")
def validate(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
) -> None:
    """Validate the configuration and show warnings."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("Configuration Validation")
    warnings = loader.validate(config)

    if not warnings:
        print_success("Configuration is valid!")
    else:
        print_warning(f"Found {len(warnings)} issue(s):")
        for w in warnings:
            console.print(f"  [yellow]- {w}[/yellow]")


@config_app.command("profile")
def profile(
    action: str = typer.Argument("list", help="list, apply, or show"),
    profile_name: Optional[str] = typer.Argument(None, help="Profile name"),
) -> None:
    """Manage configuration profiles."""
    if action == "list":
        print_header("Available Profiles")
        rows = []
        for name in PROFILES:
            try:
                cfg = PROFILES[name]()
                rows.append([
                    name,
                    cfg.chat_model,
                    cfg.embedding_model,
                    cfg.query.default_method,
                ])
            except Exception:
                rows.append([name, "?", "?", "?"])

        print_table("Profiles", ["Name", "Chat Model", "Embedding", "Query Method"], rows)

    elif action == "apply":
        if not profile_name:
            print_error("Specify a profile name: graphrag-kg config profile apply fast")
            raise typer.Exit(code=1)

        if profile_name not in PROFILES:
            print_error(f"Unknown profile '{profile_name}'. Available: {list(PROFILES.keys())}")
            raise typer.Exit(code=1)

        loader = ConfigLoader()
        config = loader.load_profile(profile_name)
        print_success(f"Applied '{profile_name}' profile")
        print_info(f"Chat model: {config.chat_model}")
        print_info(f"Query method: {config.query.default_method}")

    elif action == "show":
        if not profile_name:
            print_error("Specify a profile name")
            raise typer.Exit(code=1)
        if profile_name not in PROFILES:
            print_error(f"Unknown profile '{profile_name}'")
            raise typer.Exit(code=1)

        config = PROFILES[profile_name]()
        console.print(f"[bold]{profile_name} profile:[/bold]")
        console.print(f"  Chat: {config.chat_model}")
        console.print(f"  Embedding: {config.embedding_model}")
        console.print(f"  Query: {config.query.default_method}")
        console.print(f"  Workers: {config.pipeline.max_workers}")
