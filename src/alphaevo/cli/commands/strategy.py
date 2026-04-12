"""Strategy CLI sub-commands."""

from __future__ import annotations

import difflib
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from alphaevo.cli._helpers import _get_config, _get_store

strategy_app = typer.Typer(help="Strategy management commands")
console = Console()


@strategy_app.command("create")
def strategy_create(
    market: str = typer.Option("a_share", "--market", "-m", help="Market: a_share/us/hk"),
) -> None:
    """Interactive strategy creation from natural language (requires alphaevo[llm])."""
    console.print(Panel("🧬 Strategy Creator", style="bold cyan"))

    description = typer.prompt(
        "Describe your strategy idea",
        default="",
    )
    if not description:
        console.print("[yellow]No description provided.[/yellow]")
        raise typer.Exit(0)

    config = _get_config()
    try:
        from alphaevo.core.llm import LLMClient
        from alphaevo.strategy.generator import StrategyGenerator

        llm = LLMClient.from_config(config)
        gen = StrategyGenerator(llm)
    except Exception as e:
        console.print(f"[red]LLM setup failed: {e}[/red]")
        console.print("[dim]Install with: pip install alphaevo\\[llm][/dim]")
        raise typer.Exit(1) from None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Generating strategy...", total=None)
        try:
            strategy = gen.generate(description, market=market)
        except ValueError as e:
            console.print(f"[red]Generation failed: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            message = str(e).strip().splitlines()[0] if str(e).strip() else e.__class__.__name__
            console.print(f"[red]Generation failed: {message}[/red]")
            raise typer.Exit(1) from None

    store = _get_store(config)
    store.save(strategy)
    console.print(f"[green]✓[/green] Created: {strategy.meta.id} ({strategy.meta.name})")
    console.print(f"[dim]View with: alphaevo strategy show {strategy.meta.id}[/dim]")


@strategy_app.command("list")
def strategy_list() -> None:
    """List all strategies in the store."""
    store = _get_store()
    strategies = store.list_all()
    if not strategies:
        console.print("[yellow]No strategies found. Import some with:[/yellow]")
        console.print("  alphaevo strategy import <file.yaml>")
        console.print("  alphaevo demo  [dim]# loads builtin strategies[/dim]")
        return

    table = Table(title="📋 Strategies")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Ver", justify="right")
    table.add_column("Category")
    table.add_column("Market")
    table.add_column("Tags")
    for s in strategies:
        name = s.meta.name
        if s.meta.experimental:
            name = f"{name} ⚠️"
        table.add_row(
            s.meta.id,
            name,
            str(s.meta.version),
            s.meta.category.value if s.meta.category else "—",
            s.meta.market.value if s.meta.market else "—",
            ", ".join(s.meta.tags) if s.meta.tags else "—",
        )
    console.print(table)


@strategy_app.command("show")
def strategy_show(strategy_id: str) -> None:
    """Show strategy details."""
    store = _get_store()
    strategy = store.get(strategy_id)
    if strategy is None:
        console.print(f"[red]Strategy not found: {strategy_id}[/red]")
        raise typer.Exit(1)

    from alphaevo.strategy.dsl.serializer import StrategySerializer

    tags = ", ".join(strategy.meta.tags) if strategy.meta.tags else "—"
    parent = strategy.meta.parent_id or "—"
    experimental_note = ""
    if strategy.meta.experimental:
        experimental_note = "\n[bold yellow]⚠️  EXPERIMENTAL — uses proxy indicators, not real data feeds[/bold yellow]"
    console.print(
        Panel(
            f"[bold]{strategy.meta.name}[/bold]\n"
            f"ID: {strategy.meta.id} | Version: {strategy.meta.version}\n"
            f"Category: {strategy.meta.category.value} | Market: {strategy.meta.market.value}\n"
            f"Tags: {tags} | Parent: {parent}"
            f"{experimental_note}\n\n"
            f"[dim]{strategy.description}[/dim]",
            title="📋 Strategy Details",
            style="cyan",
        )
    )
    yaml_str = StrategySerializer().to_yaml(strategy)
    console.print(Panel(Text(yaml_str), title="YAML DSL", style="dim"))


@strategy_app.command("diff")
def strategy_diff(
    id1: str = typer.Argument(..., help="Base strategy ID"),
    id2: str = typer.Argument(..., help="Target strategy ID"),
) -> None:
    """Show structural and YAML diff between two strategies."""
    store = _get_store()
    s1 = store.get(id1)
    s2 = store.get(id2)
    if not s1 or not s2:
        missing = [i for i, s in [(id1, s1), (id2, s2)] if s is None]
        console.print(f"[red]Strategy not found: {', '.join(missing)}[/red]")
        raise typer.Exit(1)

    from alphaevo.cli.commands.analysis import _format_change_line, _strategy_diff_rows
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    console.print(Panel(f"{id1} -> {id2}", title="🧩 Strategy Diff", style="cyan"))

    diff_rows = _strategy_diff_rows(s1, s2)
    if diff_rows:
        console.print("[bold]Changed Fields[/bold]")
        for path, v1, v2 in diff_rows:
            console.print(_format_change_line(path, v1, v2))
    else:
        console.print("[dim]No structural strategy differences beyond version metadata.[/dim]")

    serializer = StrategySerializer()
    yaml1 = serializer.to_yaml(s1).splitlines()
    yaml2 = serializer.to_yaml(s2).splitlines()
    diff_lines = list(difflib.unified_diff(yaml1, yaml2, fromfile=id1, tofile=id2, lineterm=""))
    if diff_lines:
        console.print("\n[bold]📄 YAML Diff[/bold]")
        console.print(Text("\n".join(diff_lines)))


@strategy_app.command("import")
def strategy_import(
    file: str = typer.Argument(..., help="Path to strategy YAML file"),
) -> None:
    """Import a strategy from YAML file."""
    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    store = _get_store()
    try:
        strategy = store.import_from_file(path)
        console.print(f"[green]✓[/green] Imported: {strategy.meta.id} ({strategy.meta.name})")
    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")
        raise typer.Exit(1) from None


@strategy_app.command("validate")
def strategy_validate(
    file: str = typer.Argument(..., help="Path to strategy YAML file"),
    strict: bool = typer.Option(False, "--strict", help="Treat warnings as errors"),
) -> None:
    """Validate a strategy YAML file."""
    from alphaevo.strategy.dsl.parser import StrategyParseError, StrategyParser

    path = Path(file)
    if not path.exists():
        console.print(f"[red]File not found: {file}[/red]")
        raise typer.Exit(1)

    parser = StrategyParser()
    try:
        strategy = parser.parse_file(path)
        diagnostics = parser.assert_valid(strategy, strict=strict)
        console.print(f"[green]✓[/green] Valid: {strategy.meta.id} ({strategy.meta.name})")
        for w in diagnostics.warnings:
            console.print(f"  [yellow]⚠ {w}[/yellow]")
    except StrategyParseError as e:
        console.print(f"[red]✗ Invalid: {e}[/red]")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]✗ Invalid: {e}[/red]")
        raise typer.Exit(1) from None
