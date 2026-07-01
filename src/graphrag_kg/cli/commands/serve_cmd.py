"""CLI command for starting the REST API server."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from graphrag_kg.cli.utils import print_error, print_header, print_info, print_success
from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader

serve_app = typer.Typer(help="REST API server commands")


@serve_app.command("start")
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Bind address"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    config_path: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to settings.yaml"
    ),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the GraphRAG-KG REST API server."""
    loader = ConfigLoader(config_path)
    config = loader.load(config_path) if config_path else KGConfig()

    # Set global config for API
    from graphrag_kg.api.app import set_config
    set_config(config)

    print_header("GraphRAG-KG API Server")
    print_info(f"Starting server at http://{host}:{port}")
    print_info(f"API docs: http://{host}:{port}/docs")
    print_info(f"ReDoc: http://{host}:{port}/redoc")

    try:
        import uvicorn
        uvicorn.run(
            "graphrag_kg.api.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )
    except ImportError:
        print_error("uvicorn not installed. Run: pip install uvicorn[standard]")
        raise typer.Exit(code=1)
