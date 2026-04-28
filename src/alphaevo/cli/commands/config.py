"""Config CLI sub-commands."""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from alphaevo.cli._helpers import _get_config

config_app = typer.Typer(help="Configuration management")
console = Console()


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    config = _get_config()
    table = Table(title="⚙️ Configuration")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    table.add_row("LLM Model", config.llm.model)
    table.add_row("Reflect Model", config.llm.reflect_model or "(same as llm.model)")
    table.add_row("Data Adapter", config.data.adapter)
    table.add_row("DB Path", str(config.db_path))
    table.add_row("Cache Dir", str(config.data.cache_dir))
    table.add_row("Slippage", str(config.backtest.slippage))
    table.add_row("Commission", str(config.backtest.commission))
    table.add_row("Fill Policy", str(config.backtest.fill_policy))
    table.add_row("Max Evolution Rounds", str(config.evolution.max_rounds))
    table.add_row("Strategies Dir", str(config.strategies_dir))
    console.print(table)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Dotted config key (e.g. llm.model)"),
    value: str = typer.Argument(..., help="Configuration value"),
) -> None:
    """Set a configuration value and persist to user config."""
    from alphaevo.core.config import ConfigManager, _set_nested

    mgr = ConfigManager()
    overrides: dict[str, Any] = {}
    _set_nested(overrides, key, value)
    try:
        config = mgr.load(cli_overrides=overrides)
        mgr.save_user_config(config)
        console.print(f"[green]✓[/green] Set {key} = {value}")
        console.print(f"[dim]Saved to {mgr.USER_CONFIG_FILE}[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to set config: {e}[/red]")
        raise typer.Exit(1) from None
