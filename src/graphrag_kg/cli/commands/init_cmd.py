"""CLI commands for project initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from graphrag_kg.cli.utils import print_error, print_header, print_info, print_success
from graphrag_kg.core.project import ProjectManager

init_app = typer.Typer(help="Project initialization commands")


@init_app.command("init")
def init_project(
    name: str = typer.Option(
        "my-knowledge-graph", "--name", "-n",
        help="Project name",
    ),
    description: str = typer.Option(
        "", "--description", "-d",
        help="Project description",
    ),
    root: Optional[Path] = typer.Option(
        None, "--root", "-r",
        help="Project root directory (default: current directory)",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Overwrite existing project files",
    ),
) -> None:
    """Initialize a new GraphRAG-KG project."""
    project_dir = (root or Path.cwd()).resolve()

    print_header(f"Initializing GraphRAG-KG Project: {name}")

    pm = ProjectManager(project_dir)
    config = pm.init(project_name=name, description=description, force=force)

    print_success(f"Project created at: {project_dir}")
    print_info(f"Settings: {project_dir / 'settings.yaml'}")
    print_info(f"Docker Compose: {project_dir / 'docker-compose.yml'}")
    print_info(f"Environment: {project_dir / '.env'}")

    print_header("Next Steps")
    print("  1. Edit .env with your API keys")
    print("  2. Start Neo4j: docker-compose up -d")
    print("  3. Add documents to the documents/ directory")
    print("  4. Run: graphrag-kg ingest")
    print("  5. Run: graphrag-kg index")
    print("  6. Run: graphrag-kg graph sync")
    print("  7. Run: graphrag-kg query \"Your question here\"")
