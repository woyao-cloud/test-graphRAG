"""Main CLI entry point for GraphRAG-KG.

Usage:
    graphrag-kg --help
    graphrag-kg data generate --scenario pharma_supply_chain
    graphrag-kg data ground-truth
    graphrag-kg data list-scenarios
"""

from __future__ import annotations

import typer

from graphrag_kg import __version__
from graphrag_kg.cli.commands.data_cmd import data_app
from graphrag_kg.cli.commands.ingest_cmd import ingest_app
from graphrag_kg.cli.commands.init_cmd import init_project
from graphrag_kg.cli.commands.index_cmd import index_app
from graphrag_kg.cli.commands.graph_cmd import graph_app
from graphrag_kg.cli.commands.query_cmd import query_app
from graphrag_kg.cli.commands.config_cmd import config_app
from graphrag_kg.cli.commands.serve_cmd import serve_app

app = typer.Typer(
    name="graphrag-kg",
    help="GraphRAG Knowledge Graph Q&A System",
    add_completion=False,
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(data_app, name="data", help="Test data generation and ground truth management")
app.command("init", help="Initialize a new GraphRAG-KG project")(init_project)
app.add_typer(ingest_app, name="ingest", help="Document ingestion commands")
app.add_typer(index_app, name="index", help="Knowledge graph indexing commands")
app.add_typer(graph_app, name="graph", help="Neo4j graph management commands")
app.add_typer(query_app, name="query", help="Knowledge graph Q&A queries")
app.add_typer(config_app, name="config", help="Configuration management")
app.add_typer(serve_app, name="serve", help="Start the REST API server")

# Future commands (Phase 5):
# app.add_typer(serve_app, name="serve")


@app.callback()
def main_callback(
    version: bool = typer.Option(
        False, "--version", "-V", help="Show version and exit"
    ),
) -> None:
    """GraphRAG Knowledge Graph Q&A System with Neo4j + LanceDB hybrid storage."""
    if version:
        typer.echo(f"graphrag-kg v{__version__}")


if __name__ == "__main__":
    app()
