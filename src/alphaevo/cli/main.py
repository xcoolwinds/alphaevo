"""AlphaEvo CLI — Self-Evolving Stock Agent command line interface.

Thin assembler that registers sub-command modules.
"""

from __future__ import annotations

from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel

# ── App ────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="alphaevo",
    help="🧬 AlphaEvo — Self-Evolving Stock Strategy Research Agent",
    no_args_is_help=True,
)
console = Console()

# ── Shared helpers (kept importable for backward compat with tests) ───

from alphaevo.cli._helpers import _get_config, _get_store  # noqa: E402

# ── Sub-command registration ──────────────────────────────────────────
from alphaevo.cli.commands.analysis import (  # noqa: E402
    compare_command,
    leaderboard_command,
    tree_command,
)
from alphaevo.cli.commands.config import config_app  # noqa: E402
from alphaevo.cli.commands.evolution import (  # noqa: E402
    evolve_command,
    evolve_curriculum_command,
    evolve_islands_command,
    run_command,
)
from alphaevo.cli.commands.factor import factor_app  # noqa: E402
from alphaevo.cli.commands.strategy import strategy_app  # noqa: E402

app.add_typer(strategy_app, name="strategy")
app.add_typer(factor_app, name="factor")
app.add_typer(config_app, name="config")

# Top-level commands delegated to command modules
app.command("run")(run_command)
app.command("evolve")(evolve_command)
app.command("evolve-islands")(evolve_islands_command)
app.command("evolve-curriculum")(evolve_curriculum_command)
app.command("leaderboard")(leaderboard_command)
app.command("compare")(compare_command)
app.command("tree")(tree_command)

# ── Init ───────────────────────────────────────────────────────────────

_CORE_DATA_ADAPTERS = ["yfinance", "akshare"]
_OPTIONAL_DATA_BRIDGES = ["dsa"]
_KNOWN_DATA_ADAPTERS = [*_CORE_DATA_ADAPTERS, *_OPTIONAL_DATA_BRIDGES]
_LLM_MODELS = [
    "gemini/gemini-2.0-flash",
    "deepseek/deepseek-chat",
    "openai/gpt-4o-mini",
]


@app.command()
def init() -> None:
    """Interactive first-time setup — configure data source, LLM, and directories."""
    from alphaevo.core.config import ConfigManager

    console.print(Panel("🧬 AlphaEvo Setup", style="bold cyan"))

    mgr = ConfigManager()
    overrides: dict[str, Any] = {}

    # 1. Data adapter
    console.print("\n[bold]1. Data source[/bold]")
    console.print("  Core adapters: yfinance (default), akshare (A-share)")
    console.print(
        "  Optional bridge: dsa [dim](daily_stock_analysis enhancement, requires ALPHAEVO_DSA_PATH)[/dim]"
    )
    adapter = typer.prompt("  Choose data adapter", default="yfinance")
    if adapter not in _KNOWN_DATA_ADAPTERS:
        console.print(f"[yellow]Warning: '{adapter}' is not a known adapter, using anyway[/yellow]")
    elif adapter in _OPTIONAL_DATA_BRIDGES:
        console.print(
            "[dim]Using dsa as an optional external bridge, not a core built-in data source.[/dim]"
        )
    overrides.setdefault("data", {})["adapter"] = adapter

    # 2. LLM model (optional)
    console.print(
        "\n[bold]2. LLM model[/bold]"
        " [dim](for strategy generation/evolution, skip if not needed)[/dim]"
    )
    console.print("  Suggestions: " + ", ".join(_LLM_MODELS))
    model = typer.prompt("  LLM model (enter to skip)", default="")
    if model:
        overrides.setdefault("llm", {})["model"] = model

    # 3. Save
    config = mgr.load(cli_overrides=overrides)
    mgr.ensure_dirs(config)
    mgr.save_user_config(config)

    # 4. Initialize database
    from alphaevo.strategy.store import StrategyStore

    StrategyStore(config.db_path)

    console.print(f"\n[green]✓[/green] Config saved to {mgr.USER_CONFIG_FILE}")
    console.print(f"[green]✓[/green] Database at {config.db_path}")
    console.print(f"[green]✓[/green] Cache dir at {config.data.cache_dir}")
    console.print("\n[dim]Run 'alphaevo demo' to try it out![/dim]")


# ── Version ────────────────────────────────────────────────────────────


@app.command()
def version() -> None:
    """Show AlphaEvo version."""
    from alphaevo import __version__

    console.print(f"AlphaEvo v{__version__}")


# ── Demo ───────────────────────────────────────────────────────────────


@app.command()
def demo(
    real: bool = typer.Option(False, "--real", help="Use real market data (requires network)"),
    market: str = typer.Option("us", "--market", help="Market for real data: us or cn"),
) -> None:
    """Run the showcase demo — synthetic by default, or `--real` with live market data."""
    if real:
        from alphaevo.cli.demo import run_real_demo

        run_real_demo(console, market=market)
    else:
        from alphaevo.cli.demo import run_demo

        run_demo(console)


# ── Backward-compat re-exports (used by tests) ────────────────────────

from alphaevo.cli.commands.factor import (  # noqa: E402, F811
    _build_forward_returns,
    _load_factor_history,
)

__all__ = [
    "app",
    "_get_config",
    "_get_store",
    "_load_factor_history",
    "_build_forward_returns",
]


if __name__ == "__main__":
    app()
