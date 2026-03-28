"""Verbose logging utilities using Rich library."""

import json
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax


class VerboseLogger:
    """Handles verbose output at different levels."""

    def __init__(self, level: int = 0, console: Console | None = None):
        """
        Initialize logger with verbosity level.

        Args:
            level: 0=silent, 1=major actions, 2=detailed, 3=debug
            console: Optional Rich Console instance (default: new Console())
        """
        self.level = level
        self._console = console or Console()

    def _should_log(self, min_level: int) -> bool:
        """Check if current verbosity level should log."""
        return self.level >= min_level

    def action(self, action: str, details: str = "") -> None:
        """Log major action (level 1+)."""
        if not self._should_log(1):
            return

        self._console.print(f"[bold blue]→[/bold blue] {action}", end="")
        if details:
            self._console.print(f" [dim]{details}[/dim]")
        else:
            self._console.print()

    def detail(self, key: str, value: Any) -> None:
        """Log detailed information (level 2+)."""
        if not self._should_log(2):
            return

        self._console.print(f"  [cyan]•[/cyan] {key}: [yellow]{value}[/yellow]")

    def debug(self, title: str, data: Any, syntax: str = "json") -> None:
        """Log debug information (level 3+)."""
        if not self._should_log(3):
            return

        if isinstance(data, (dict, list)):
            data_str = json.dumps(data, indent=2, ensure_ascii=False)
        else:
            data_str = str(data)

        # Truncate if too long
        if len(data_str) > 2000:
            data_str = data_str[:2000] + "\n... (truncated)"

        self._console.print(
            Panel(
                Syntax(data_str, syntax, theme="monokai", word_wrap=True),
                title=f"[bold magenta]{title}[/bold magenta]",
                border_style="magenta",
                expand=False,
            )
        )

    def success(self, message: str) -> None:
        """Log success message (level 1+)."""
        if not self._should_log(1):
            return

        self._console.print(f"[bold green]✓[/bold green] {message}")

    def warning(self, message: str) -> None:
        """Log warning (level 1+)."""
        if not self._should_log(1):
            return

        self._console.print(f"[bold yellow]⚠[/bold yellow] {message}")

    def error(self, message: str) -> None:
        """Log error (always shown)."""
        self._console.print(f"[bold red]✗[/bold red] {message}", style="bold red")
