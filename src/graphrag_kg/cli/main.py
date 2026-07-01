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

app = typer.Typer(
    name="graphrag-kg",
    help="GraphRAG Knowledge Graph Q&A System",
    add_completion=False,
    no_args_is_help=True,
)

# Register subcommands
app.add_typer(data_app, name="data", help="Test data generation and ground truth management")

# Future commands (Phases 1-5):
# app.add_typer(init_app, name="init")
# app.add_typer(ingest_app, name="ingest")
# app.add_typer(index_app, name="index")
# app.add_typer(graph_app, name="graph")
# app.add_typer(query_app, name="query")
# app.add_typer(config_app, name="config")
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
