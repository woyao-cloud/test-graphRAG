"""CLI formatting utilities using Rich."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax

console = Console(force_terminal=True)


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[bold green][OK][/bold green] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[bold red][ERROR][/bold red] {message}")


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[bold blue][INFO][/bold blue] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[bold yellow][WARN][/bold yellow] {message}")


def print_header(title: str) -> None:
    """Print a section header."""
    console.print()
    console.print(Panel(f"[bold]{title}[/bold]", border_style="blue"))
    console.print()


def print_table(title: str, columns: list[str], rows: list[list[str]]) -> None:
    """Print a formatted table."""
    table = Table(title=title, show_header=True, header_style="bold")
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(table)


def print_json_syntax(data: str) -> None:
    """Print JSON with syntax highlighting."""
    syntax = Syntax(data, "json", theme="monokai", line_numbers=False)
    console.print(syntax)


def create_progress() -> Progress:
    """Create a Rich progress bar for long-running tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
