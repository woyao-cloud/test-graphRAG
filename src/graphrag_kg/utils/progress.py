"""Progress bar and callback utilities for pipeline observability."""

from __future__ import annotations

from typing import Any, Callable, Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_pipeline_progress() -> Progress:
    """Create a Rich progress bar configured for pipeline steps."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        transient=False,
    )


class ProgressCallback:
    """Wraps a Rich Progress instance for pipeline step reporting.

    Usage:
        with create_pipeline_progress() as progress:
            cb = ProgressCallback(progress)
            task_id = cb.start_task("Extracting entities", total=100)
            for i in range(100):
                cb.update(task_id, advance=1)
            cb.complete_task(task_id)
    """

    def __init__(self, progress: Progress):
        self.progress = progress
        self._tasks: dict[str, TaskID] = {}

    def start_task(
        self, description: str, total: int = 100, **kwargs: Any
    ) -> str:
        """Start a new progress task. Returns a task key."""
        task_id = self.progress.add_task(
            f"[cyan]{description}[/cyan]", total=total, **kwargs
        )
        self._tasks[description] = task_id
        return description

    def update(self, task_key: str, advance: int = 1, **kwargs: Any) -> None:
        """Update progress for a task."""
        if task_key in self._tasks:
            self.progress.update(self._tasks[task_key], advance=advance, **kwargs)

    def complete_task(self, task_key: str) -> None:
        """Mark a task as complete."""
        if task_key in self._tasks:
            task_id = self._tasks[task_key]
            self.progress.update(task_id, completed=self.progress.tasks[task_id].total)
            self.progress.remove_task(task_id)
            del self._tasks[task_key]

    def set_description(self, task_key: str, description: str) -> None:
        """Update task description."""
        if task_key in self._tasks:
            self.progress.update(self._tasks[task_key], description=description)


def track_progress(
    items: list[Any],
    description: str = "Processing",
    callback: Optional[Callable[[int, Any], None]] = None,
) -> Progress:
    """Create a progress bar tracking iteration over items.

    Usage:
        with track_progress(files, "Ingesting documents") as progress:
            for file in files:
                process(file)
                progress.advance(progress.task_ids[0])
    """
    progress = Progress(
        TextColumn("[bold blue]{task.description}[/bold blue]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
    )
    progress.add_task(description, total=len(items))
    return progress
