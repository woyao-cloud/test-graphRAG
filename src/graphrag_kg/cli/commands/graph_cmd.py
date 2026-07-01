"""CLI commands for Neo4j graph management."""

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
    print_table,
)
from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.graph.connection import Neo4jConnection
from graphrag_kg.graph.queries import CypherQueries
from graphrag_kg.graph.sync import Neo4jGraphSync
from graphrag_kg.core.errors import Neo4jConnectionError

graph_app = typer.Typer(help="Neo4j graph management commands")


@graph_app.command("sync")
def sync(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
    clear_first: bool = typer.Option(
        False, "--clear", help="Clear Neo4j database before syncing",
    ),
) -> None:
    """Sync indexed data from Parquet to Neo4j."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("Syncing to Neo4j")

    syncer = Neo4jGraphSync(config)
    try:
        results = syncer.sync_all(clear_first=clear_first)
        print_success("Sync complete!")
        print_table("Sync Results", ["Type", "Count"], [
            ["Entities", str(results.get("entities", 0))],
            ["Relationships", str(results.get("relationships", 0))],
            ["Communities", str(results.get("communities", 0))],
            ["Documents", str(results.get("documents", 0))],
            ["Text Units", str(results.get("text_units", 0))],
        ])
    except Exception as e:
        print_error(f"Sync failed: {e}")
        raise typer.Exit(code=1)


@graph_app.command("status")
def status(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
) -> None:
    """Show Neo4j graph statistics."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    print_header("Neo4j Graph Status")

    conn = Neo4jConnection(config.neo4j)
    try:
        conn.connect()
        health = conn.health_check()
        if health["connected"]:
            print_success(f"Connected to Neo4j {health.get('neo4j_version', '')}")
        else:
            print_error(f"Not connected: {health.get('error', 'unknown')}")
            raise typer.Exit(code=1)

        with conn.session() as session:
            queries = CypherQueries(session)
            stats = queries.get_graph_stats()

            print_table("Graph Statistics", ["Metric", "Value"], [
                ["Total Nodes", str(stats["total_nodes"])],
                ["Entities", str(stats["entities"])],
                ["Relationships", str(stats["relationships"])],
                ["Communities", str(stats["communities"])],
                ["Documents", str(stats["documents"])],
            ])

            # Top entities
            top = queries.get_top_entities(5)
            if top:
                print_header("Top Entities by Degree")
                rows = [[e["name"], e["type"], str(e["degree"])] for e in top]
                print_table("", ["Name", "Type", "Degree"], rows)

    except Neo4jConnectionError as e:
        print_warning(f"Neo4j not available: {e}")
        print_info("Start Neo4j with: docker-compose up -d")
    finally:
        conn.close()


@graph_app.command("drop")
def drop(
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c",
        help="Path to settings.yaml",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete all nodes and relationships from Neo4j."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    if not force:
        confirmed = typer.confirm(
            "This will delete ALL nodes and relationships in Neo4j. Continue?"
        )
        if not confirmed:
            print_info("Aborted.")
            raise typer.Exit()

    print_header("Clearing Neo4j Database")
    conn = Neo4jConnection(config.neo4j)
    try:
        conn.connect()
        result = conn.clear_database(confirm=True)
        print_success(f"Deleted {result['nodes_deleted']} nodes, "
                      f"{result['relationships_deleted']} relationships")
    except Neo4jConnectionError as e:
        print_error(f"Neo4j not available: {e}")
    finally:
        conn.close()
