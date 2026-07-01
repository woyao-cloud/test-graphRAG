"""Pipeline progress monitoring with Rich console output.

Provides callbacks that hook into the graphrag indexing pipeline
to display real-time progress with formatted output.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

console = Console()


class PipelineMonitor:
    """Real-time progress monitor for graphrag indexing pipeline.

    Displays:
    - Current workflow step
    - Elapsed time per step
    - Entity/relationship counts (when available)
    - Token usage estimates
    - Error notifications
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.start_time = time.time()
        self.step_times: dict[str, float] = {}
        self._current_step: Optional[str] = None
        self._step_start: float = 0.0

    # ------------------------------------------------------------------
    # Callback Interface
    # ------------------------------------------------------------------

    def on_progress(self, event: str, data: dict[str, Any]) -> None:
        """Main callback for pipeline events.

        Args:
            event: Event type — 'starting', 'running', 'step_start',
                   'step_complete', 'statistics', 'complete', 'error'.
            data: Event-specific data dict.
        """
        if event == "starting":
            self._on_start(data)
        elif event == "step_start":
            self._on_step_start(data)
        elif event == "step_complete":
            self._on_step_complete(data)
        elif event == "statistics":
            self._on_statistics(data)
        elif event == "complete":
            self._on_complete(data)
        elif event == "error":
            self._on_error(data)
        elif self.verbose:
            console.print(f"  [dim]{event}: {data}[/dim]")

    def _on_start(self, data: dict[str, Any]) -> None:
        """Pipeline starting."""
        method = data.get("method", "standard")
        workflows = data.get("workflows", 0)
        console.print()
        console.print(Panel(
            f"[bold]Starting GraphRAG Indexing Pipeline[/bold]\n"
            f"Method: [cyan]{method}[/cyan] | "
            f"Workflows: [cyan]{workflows}[/cyan]",
            border_style="blue",
        ))
        console.print()

    def _on_step_start(self, data: dict[str, Any]) -> None:
        """A workflow step has started."""
        name = data.get("step", "unknown")
        self._current_step = name
        self._step_start = time.time()
        console.print(f"  [bold yellow]→[/bold yellow] {name}...", end="")

    def _on_step_complete(self, data: dict[str, Any]) -> None:
        """A workflow step has completed."""
        name = data.get("step", self._current_step or "unknown")
        elapsed = time.time() - self._step_start if self._step_start else 0
        self.step_times[name] = elapsed
        self._current_step = None

        status = "[bold green]OK[/bold green]"
        if data.get("warnings"):
            status = "[bold yellow]WARN[/bold yellow]"

        console.print(f" {status} ({elapsed:.1f}s)")

        if self.verbose and data:
            for key, value in data.items():
                if key not in ("step", "warnings"):
                    console.print(f"      [dim]{key}: {value}[/dim]")

    def _on_statistics(self, data: dict[str, Any]) -> None:
        """Extracted statistics available."""
        console.print()
        console.print(Panel(
            "[bold]Indexing Statistics[/bold]",
            border_style="green",
        ))
        for key, value in data.items():
            console.print(f"  [bold]{key}[/bold]: {value}")

    def _on_complete(self, data: dict[str, Any]) -> None:
        """Pipeline complete."""
        total_time = time.time() - self.start_time
        console.print()
        console.print(Panel(
            f"[bold green]Indexing Complete[/bold green]\n"
            f"Total time: [cyan]{total_time:.1f}s[/cyan]",
            border_style="green",
        ))

        if self.step_times:
            console.print()
            console.print("[bold]Step Timing:[/bold]")
            for step, elapsed in self.step_times.items():
                bar = "█" * min(int(elapsed / max(self.step_times.values()) * 20), 20)
                console.print(f"  {step:40s} {bar} {elapsed:.1f}s")

    def _on_error(self, data: dict[str, Any]) -> None:
        """Pipeline error."""
        error_msg = data.get("error", "Unknown error")
        console.print()
        console.print(f"[bold red]ERROR:[/bold red] {error_msg}")


class SilentMonitor(PipelineMonitor):
    """Non-verbose monitor that only prints on completion or error."""

    def __init__(self):
        super().__init__(verbose=False)

    def _on_step_start(self, data: dict[str, Any]) -> None:
        pass

    def _on_step_complete(self, data: dict[str, Any]) -> None:
        name = data.get("step", self._current_step or "unknown")
        elapsed = time.time() - self._step_start if self._step_start else 0
        self.step_times[name] = elapsed
        self._current_step = None


class ProgressBarMonitor(PipelineMonitor):
    """Monitor with Rich progress bars for each workflow step."""

    def __init__(self):
        super().__init__(verbose=False)
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/bold blue]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self._task_id: Optional[TaskID] = None

    def _on_start(self, data: dict[str, Any]) -> None:
        super()._on_start(data)
        self._progress.start()
        self._task_id = self._progress.add_task(
            "Indexing...", total=data.get("workflows", 10)
        )

    def _on_step_complete(self, data: dict[str, Any]) -> None:
        super()._on_step_complete(data)
        if self._task_id is not None:
            self._progress.advance(self._task_id)

    def _on_complete(self, data: dict[str, Any]) -> None:
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=self._progress.tasks[self._task_id].total)
        self._progress.stop()
        super()._on_complete(data)

    def _on_error(self, data: dict[str, Any]) -> None:
        self._progress.stop()
        super()._on_error(data)
