"""Rich terminal renderer for research events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from alphaevo.research_log.logger import ResearchEvent, ResearchLogger

_EVENT_STYLES: dict[str, str] = {
    "hypothesis": "bold cyan",
    "observation": "dim white",
    "diagnosis": "bold yellow",
    "insight": "bold green",
    "decision": "bold magenta",
    "experiment": "blue",
    "result": "bold white",
    "reflection": "italic yellow",
}

_EVENT_ICONS: dict[str, str] = {
    "hypothesis": "💡",
    "observation": "👁️",
    "diagnosis": "🔍",
    "insight": "🧠",
    "decision": "✅",
    "experiment": "🧪",
    "result": "📊",
    "reflection": "🪞",
}


def render_event(event: ResearchEvent, console: Console | None = None) -> None:
    """Print a single research event to the terminal."""
    console = console or Console()
    icon = _EVENT_ICONS.get(event.event_type, "📌")
    style = _EVENT_STYLES.get(event.event_type, "white")
    prefix = f"[dim]R{event.round_num}[/dim] " if event.round_num > 0 else ""
    console.print(f"  {prefix}{icon} [{style}]{event.content}[/{style}]")


def render_round_header(round_num: int, strategy_id: str, console: Console | None = None) -> None:
    """Print a round header."""
    console = console or Console()
    console.print(
        Panel(
            f"[bold]Round {round_num}[/bold] — {strategy_id}",
            style="blue",
            width=60,
        )
    )


def render_log_summary(logger: ResearchLogger, console: Console | None = None) -> None:
    """Render the full research log as a Rich panel."""
    console = console or Console()
    if not logger.events:
        console.print("[dim]No research events recorded.[/dim]")
        return

    lines = []
    current_round = -1
    for event in logger.events:
        if event.round_num != current_round:
            current_round = event.round_num
            if current_round > 0:
                lines.append(f"\n[bold blue]── Round {current_round} ──[/bold blue]")
        icon = _EVENT_ICONS.get(event.event_type, "📌")
        style = _EVENT_STYLES.get(event.event_type, "white")
        lines.append(f"  {icon} [{style}]{event.content}[/{style}]")

    text = "\n".join(lines)
    console.print(Panel(text, title="🔬 Research Log", border_style="green"))
