"""Evolution CLI commands — run, evolve, evolve-islands, evolve-curriculum."""

from __future__ import annotations

import asyncio
import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from alphaevo.cli._helpers import _get_config

logger = logging.getLogger(__name__)
console = Console()


def _write_strategy_yaml_artifacts(
    result: object,
    *,
    strategy_id: str,
    out_path: Path,
) -> tuple[Path | None, Path | None, int]:
    """Export per-round strategy YAMLs plus a champion shortcut file."""
    from alphaevo.strategy.dsl.serializer import StrategySerializer

    serializer = StrategySerializer()
    strategy_dir: Path | None = None
    champion_path: Path | None = None
    champion_id = getattr(result, "champion_id", None)
    written: set[str] = set()

    for round_result in getattr(result, "rounds", []):
        strategy = getattr(round_result, "strategy", None)
        meta = getattr(strategy, "meta", None)
        round_strategy_id = getattr(meta, "id", None)
        if strategy is None or not round_strategy_id or round_strategy_id in written:
            continue
        if strategy_dir is None:
            strategy_dir = out_path / f"{strategy_id}_strategies"
            strategy_dir.mkdir(parents=True, exist_ok=True)
        try:
            serializer.to_file(strategy, strategy_dir / f"{round_strategy_id}.yaml")
        except Exception:
            continue
        written.add(round_strategy_id)
        if round_strategy_id == champion_id:
            champion_path = out_path / f"{strategy_id}_champion.yaml"
            try:
                serializer.to_file(strategy, champion_path)
            except Exception:
                champion_path = None

    return strategy_dir, champion_path, len(written)


def _format_applied_change_line(
    change_type: str, target: str, from_value: object, to_value: object
) -> Text:
    """Render a mutation line without Rich eating target selectors."""
    from alphaevo.reflection.mutator import _sanitize_condition_value

    line = Text("    • ")
    line.append(change_type)
    line.append(" on ")
    line.append(str(target), style="dim")
    line.append(" (")
    line.append(str(_sanitize_condition_value(from_value)))
    line.append(" -> ")
    line.append(str(_sanitize_condition_value(to_value)))
    line.append(")")
    return line


def run_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to run"),
    samples: int = typer.Option(60, "--samples", "-n", help="Max stock samples"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    report_dir: str = typer.Option("reports", "--output", "-o", help="Report output directory"),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="Data adapter override; core: yfinance/akshare, optional bridge: dsa",
    ),
    sampling: str | None = typer.Option(None, "--sampling", help="Sampling method"),
    wf_folds: int | None = typer.Option(None, "--wf-folds", help="Walk-forward folds"),
    wf_train_pct: float | None = typer.Option(None, "--wf-train-pct", help="Walk-forward train %"),
    wf_pass_gap: float | None = typer.Option(None, "--wf-pass-gap", help="Walk-forward pass gap"),
    stress_window_days: int | None = typer.Option(
        None, "--stress-window-days", help="Stress window days"
    ),
    stress_window_top_k: int | None = typer.Option(
        None, "--stress-window-top-k", help="Stress window top-k"
    ),
) -> None:
    """Run the full research loop: sample → backtest → evaluate → report."""
    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    bt_overrides: dict = {}
    if wf_folds is not None:
        bt_overrides["walk_forward_folds"] = wf_folds
    if wf_train_pct is not None:
        bt_overrides["walk_forward_train_pct"] = wf_train_pct
    if wf_pass_gap is not None:
        bt_overrides["walk_forward_pass_gap"] = wf_pass_gap
    if stress_window_days is not None:
        bt_overrides["stress_window_days"] = stress_window_days
    if stress_window_top_k is not None:
        bt_overrides["stress_window_top_k"] = stress_window_top_k
    if bt_overrides:
        overrides["backtest"] = bt_overrides

    config = _get_config(overrides or None)

    from alphaevo.models.enums import SamplingMethod

    sampling_method: SamplingMethod | None = None
    if sampling:
        try:
            sampling_method = SamplingMethod(sampling)
        except ValueError:
            console.print(f"[red]Unknown sampling method: {sampling}[/red]")
            raise typer.Exit(1) from None

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        if s > e:
            console.print("[red]Error: --start date must be before --end date[/red]")
            raise typer.Exit(1)
        dr = (s, e)
    elif end_date:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"🚀 Running research loop for [cyan]{strategy_id}[/cyan]",
            style="bold green",
        )
    )

    from alphaevo.orchestrator.pipeline import RunPipeline

    pipeline = RunPipeline(config)
    pipeline.ensure_builtin_strategies()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)

        run_kwargs: dict = {
            "max_symbols": samples,
            "date_range": dr,
            "report_dir": Path(report_dir),
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            run_kwargs["sampling_method"] = sampling_method

        try:
            result = asyncio.run(pipeline.run(strategy_id, **run_kwargs))
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1) from None
        except Exception as e:
            logger.exception("Pipeline failed")
            console.print(f"[red]Pipeline failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    ev = result.evaluation
    table = Table(title=f"📊 Results: {strategy_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Win Rate", f"{ev.overall.win_rate:.1%}")
    table.add_row("Avg Return", f"{ev.overall.avg_return:.2%}")
    table.add_row("P/L Ratio", f"{ev.overall.profit_loss_ratio:.2f}")
    table.add_row("Max Drawdown", f"{ev.overall.max_drawdown:.1%}")
    table.add_row("Sharpe Ratio", f"{ev.overall.sharpe_ratio:.2f}")
    table.add_row("Total Signals", str(ev.overall.signal_count))
    table.add_row("Avg Holding Days", f"{ev.overall.avg_holding_days:.1f}")
    table.add_row("Total Return", f"{ev.overall.total_return:.2%}")
    table.add_row("Confidence Score", f"[bold green]{ev.confidence_score:.1%}[/bold green]")
    console.print(table)

    # Equity curve and charts (plotext)
    if ev.overall.signal_count > 0:
        from alphaevo.evaluator.reporter import Reporter

        equity_chart = Reporter.plot_equity_curve(
            result.backtest_result.signals,
            title=f"📈 Equity Curve — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{equity_chart}"))

        dist_chart = Reporter.plot_return_distribution(
            result.backtest_result.signals,
            title=f"📊 Return Distribution — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{dist_chart}"))

        dd_chart = Reporter.plot_drawdown_curve(
            result.backtest_result.signals,
            title=f"📉 Drawdown — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{dd_chart}"))

    # Portfolio-level simulation
    if ev.overall.signal_count > 0:
        try:
            from alphaevo.backtest.portfolio import PortfolioBacktester, PortfolioConfig

            pf = PortfolioBacktester(PortfolioConfig())
            pf_result = pf.simulate(result.backtest_result)
            if pf_result.total_trades > 0:
                pf_table = Table(title="💼 Portfolio Simulation ($100K, max 5 positions)")
                pf_table.add_column("Metric", style="cyan")
                pf_table.add_column("Value", justify="right")
                pf_table.add_row("Final Equity", f"${pf_result.final_equity:,.0f}")
                pf_table.add_row("Total Return", f"{pf_result.total_return:.2%}")
                pf_table.add_row("Max Drawdown", f"{pf_result.max_drawdown:.1%}")
                pf_table.add_row("Sharpe Ratio", f"{pf_result.sharpe_ratio:.2f}")
                pf_table.add_row("Max Concurrent", str(pf_result.max_concurrent_positions))
                pf_table.add_row("Capital Utilization", f"{pf_result.capital_utilization:.0%}")
                pf_table.add_row("Win Rate", f"{pf_result.win_rate:.1%}")
                console.print(pf_table)
        except Exception:
            pass  # Portfolio simulation is optional

    if ev.overall.signal_count == 0:
        console.print(
            "\n[yellow]⚠ No trades fired in this run.[/yellow] "
            "That usually means the strategy was too strict for the sampled window,"
            " not that the backtest crashed."
        )
        console.print(
            f"[dim]Adapter: {config.data.adapter} | "
            f"Strategy market: {result.strategy.meta.market.value} | "
            f"Sampled symbols: {len(result.batch.symbols)}[/dim]"
        )
        if result.strategy.meta.market.value == "a_share" and config.data.adapter == "yfinance":
            console.print(
                "[dim]Hint: A-share strategies usually work better with "
                "`--adapter akshare` than the default yfinance adapter.[/dim]"
            )
        console.print(
            "[dim]Hint: try a longer date range, `--sampling strategy_scoped`, "
            "or evolve the strategy to relax strict thresholds.[/dim]"
        )

    if result.report_path:
        console.print(f"\n📄 Report saved to: {result.report_path}")


def evolve_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to evolve"),
    rounds: int = typer.Option(3, "--rounds", "-r", help="Evolution rounds"),
    method: str = typer.Option("hybrid", "--method", "-m", help="llm/param_search/hybrid"),
    samples: int = typer.Option(60, "--samples", "-n", help="Max stock samples per round"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    adapter: str | None = typer.Option(
        None,
        "--adapter",
        help="Data adapter override; core: yfinance/akshare, optional bridge: dsa",
    ),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
    reflect_model: str | None = typer.Option(
        None, "--reflect-model", help="Reflection model override"
    ),
    sampling: str | None = typer.Option(None, "--sampling", help="Sampling method"),
    output_dir: str = typer.Option(
        "reports", "--output", "-o", help="Output directory for artifacts"
    ),
    wf_folds: int | None = typer.Option(None, "--wf-folds", help="Walk-forward folds"),
    wf_train_pct: float | None = typer.Option(None, "--wf-train-pct", help="Walk-forward train %"),
    wf_pass_gap: float | None = typer.Option(None, "--wf-pass-gap", help="Walk-forward pass gap"),
    stress_window_days: int | None = typer.Option(
        None, "--stress-window-days", help="Stress window days"
    ),
    stress_window_top_k: int | None = typer.Option(
        None, "--stress-window-top-k", help="Stress window top-k"
    ),
) -> None:
    """Self-evolve a strategy through multi-round improvement."""
    from alphaevo.models.enums import EvolutionMethod, SamplingMethod
    from alphaevo.orchestrator.evolution import EvolutionPipeline

    try:
        evo_method = EvolutionMethod(method)
    except ValueError:
        console.print(f"[red]Unknown method: {method}. Use: llm, param_search, hybrid[/red]")
        raise typer.Exit(1) from None

    sampling_method: SamplingMethod | None = None
    if sampling:
        try:
            sampling_method = SamplingMethod(sampling)
        except ValueError:
            console.print(f"[red]Unknown sampling method: {sampling}[/red]")
            raise typer.Exit(1) from None

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        if s > e:
            console.print("[red]Error: --start date must be before --end date[/red]")
            raise typer.Exit(1)
        dr = (s, e)
    elif end_date:
        console.print("[red]Error: --end requires --start[/red]")
        raise typer.Exit(1)

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    llm_overrides: dict = {}
    if model:
        llm_overrides["model"] = model
    if reflect_model:
        llm_overrides["reflect_model"] = reflect_model
    if llm_overrides:
        overrides["llm"] = llm_overrides
    bt_overrides: dict = {}
    if wf_folds is not None:
        bt_overrides["walk_forward_folds"] = wf_folds
    if wf_train_pct is not None:
        bt_overrides["walk_forward_train_pct"] = wf_train_pct
    if wf_pass_gap is not None:
        bt_overrides["walk_forward_pass_gap"] = wf_pass_gap
    if stress_window_days is not None:
        bt_overrides["stress_window_days"] = stress_window_days
    if stress_window_top_k is not None:
        bt_overrides["stress_window_top_k"] = stress_window_top_k
    if bt_overrides:
        overrides["backtest"] = bt_overrides

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"🧬 Evolving [cyan]{strategy_id}[/cyan] for {rounds} rounds\n"
            f"Method: {evo_method.value} | Samples: {samples}",
            style="bold magenta",
        )
    )

    pipeline = EvolutionPipeline(config)
    log_lines: list[str] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting evolution...", total=None)

        def on_progress(msg: str) -> None:
            progress.update(task, description=msg)
            log_lines.append(msg)

        evolve_kwargs: dict = {
            "rounds": rounds,
            "method": evo_method,
            "max_symbols": samples,
            "date_range": dr,
            "on_progress": on_progress,
        }
        if sampling_method is not None:
            evolve_kwargs["sampling_method"] = sampling_method

        try:
            result = pipeline.evolve(strategy_id, **evolve_kwargs)
        except Exception as e:
            logger.exception("Evolution failed")
            console.print(f"[red]Evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Summary table
    if result.rounds:
        table = Table(title=f"🧬 Evolution Results: {strategy_id}")
        table.add_column("Round", justify="right", style="bold")
        table.add_column("Strategy", style="cyan")
        table.add_column("Win Rate", justify="right")
        table.add_column("Avg Ret", justify="right")
        table.add_column("P/L", justify="right")
        table.add_column("Max DD", justify="right")
        table.add_column("Signals", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Status")

        for r in result.rounds:
            ev = r.evaluation
            status = "[green]✓ improved[/green]" if r.improved else "[dim]—[/dim]"
            table.add_row(
                str(r.round_num),
                r.strategy.meta.id,
                f"{ev.overall.win_rate:.1%}",
                f"{ev.overall.avg_return:.2%}",
                f"{ev.overall.profit_loss_ratio:.2f}",
                f"{ev.overall.max_drawdown:.1%}",
                str(ev.overall.signal_count),
                f"{ev.confidence_score:.1%}",
                status,
            )
        console.print(table)

        # Evolution score progress chart
        round_scores = [(r.strategy.meta.id, r.evaluation.confidence_score) for r in result.rounds]
        from alphaevo.evaluator.reporter import Reporter

        evo_chart = Reporter.plot_evolution_scores(
            round_scores,
            title=f"🧬 Evolution Progress — {strategy_id}",
        )
        console.print(Text.from_ansi(f"\n{evo_chart}"))

        # Equity curve for champion
        champion_round = next(
            (r for r in result.rounds if r.strategy.meta.id == result.champion_id),
            result.rounds[-1],
        )
        if champion_round.evaluation.overall.signal_count > 0:
            bt = getattr(champion_round, "backtest_result", None)
            if bt is not None and bt.signals:
                eq_chart = Reporter.plot_equity_curve(
                    bt.signals,
                    title=f"📈 Champion Equity — {result.champion_id}",
                )
                console.print(Text.from_ansi(f"\n{eq_chart}"))
    else:
        console.print("[yellow]No successful evolution rounds were recorded.[/yellow]")

    # Per-round changes detail
    has_changes = any(r.reflection and r.reflection.proposed_changes for r in result.rounds)
    if has_changes:
        console.print("\n[bold]📝 Changes Applied Per Round[/bold]")
        for r in result.rounds:
            if r.reflection and r.reflection.proposed_changes:
                console.print(f"  [cyan]Round {r.round_num}[/cyan]:")
                for ch in r.reflection.proposed_changes:
                    console.print(
                        _format_applied_change_line(
                            ch.change_type.value,
                            ch.target,
                            ch.from_value,
                            ch.to_value,
                        )
                    )
                    if ch.reason:
                        console.print(f"      [dim]{ch.reason}[/dim]")

    # Anti-overfit warnings
    if result.rounds:
        last = result.rounds[-1].evaluation
        af = last.anti_overfit
        if af.train_val_gap > 0.10:
            console.print(
                f"\n[yellow]⚠ Overfit warning: train-val gap = "
                f"{af.train_val_gap:.1%} (threshold: 10%)[/yellow]"
            )
        if af.val_test_gap > 0.08:
            console.print(
                f"[yellow]⚠ Overfit warning: val-test gap = "
                f"{af.val_test_gap:.1%} (threshold: 8%)[/yellow]"
            )

    if evo_method != EvolutionMethod.PARAM_SEARCH:
        telemetry_rows = []
        for round_result in result.rounds:
            telemetry = getattr(getattr(round_result, "reflection", None), "llm_telemetry", None)
            if telemetry is None:
                continue
            calls = getattr(telemetry, "calls", []) or []
            timeout_failures = sum(
                1
                for call in calls
                if not getattr(call, "success", True)
                and any(
                    token in str(getattr(call, "error", "")).lower()
                    for token in ("timeout", "timed out")
                )
            )
            telemetry_rows.append(
                (
                    str(round_result.round_num),
                    getattr(telemetry, "path", ""),
                    str(len(calls)),
                    f"{getattr(telemetry, 'total_duration_ms', 0)} ms",
                    str(timeout_failures),
                )
            )

        if telemetry_rows:
            telemetry_table = Table(title="🧠 LLM Runtime Telemetry")
            telemetry_table.add_column("Round", justify="right", style="bold")
            telemetry_table.add_column("Path", style="cyan")
            telemetry_table.add_column("Calls", justify="right")
            telemetry_table.add_column("Total", justify="right")
            telemetry_table.add_column("Timeouts", justify="right")
            for row in telemetry_rows:
                telemetry_table.add_row(*row)
            console.print(telemetry_table)

    if result.rounds:
        console.print(
            f"\n🏆 Champion: [cyan]{result.champion_id}[/cyan] "
            f"(score: {result.champion_score:.1%}, "
            f"improvement: {result.improvement:+.1%})"
        )
        if result.champion_id:
            console.print(
                f"[dim]View champion DSL: alphaevo strategy show {result.champion_id}[/dim]"
            )
    if result.early_stopped:
        console.print(f"[yellow]⚠ Early stopped: {result.stop_reason}[/yellow]")

    # Write output artifacts
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    from alphaevo.evaluator.reporter import Reporter

    evo_md = Reporter.evolution_report(result)
    (out_path / f"{strategy_id}_evolution.md").write_text(evo_md, encoding="utf-8")
    research_md = Reporter.research_report(result)
    (out_path / f"{strategy_id}_research_report.md").write_text(research_md, encoding="utf-8")
    if evo_method != EvolutionMethod.PARAM_SEARCH:
        llm_md = Reporter.llm_evidence_report(result)
        (out_path / f"{strategy_id}_llm_evidence.md").write_text(llm_md, encoding="utf-8")

    if hasattr(pipeline, "research_log"):
        log_md = pipeline.research_log.to_markdown()
        (out_path / f"{strategy_id}_research_log.md").write_text(log_md, encoding="utf-8")

    strategy_dir, champion_yaml, written_count = _write_strategy_yaml_artifacts(
        result,
        strategy_id=strategy_id,
        out_path=out_path,
    )

    # Export trajectory data for training (JSONL + ShareGPT + preference pairs)
    _trajectory_files: list[str] = []
    trajectory = getattr(result, "trajectory", None)
    if trajectory is not None and trajectory.steps:
        from alphaevo.research_log.trajectory import (
            export_jsonl,
            export_preference_pairs,
            export_sharegpt,
        )

        traj_dir = out_path / "trajectory"
        traj_dir.mkdir(parents=True, exist_ok=True)
        try:
            jsonl_path = export_jsonl(trajectory, traj_dir / f"{strategy_id}_trajectory.jsonl")
            _trajectory_files.append(str(jsonl_path))
        except Exception as e:
            logger.debug("JSONL export failed: %s", e)
        try:
            sharegpt_path = export_sharegpt(trajectory, traj_dir / f"{strategy_id}_sharegpt.json")
            _trajectory_files.append(str(sharegpt_path))
        except Exception as e:
            logger.debug("ShareGPT export failed: %s", e)
        try:
            pref_path = export_preference_pairs(
                trajectory, traj_dir / f"{strategy_id}_preference.json"
            )
            _trajectory_files.append(str(pref_path))
        except Exception as e:
            logger.debug("Preference pairs export failed: %s", e)

    console.print(f"\n📄 Artifacts saved to: {out_path}")
    if strategy_dir is not None and written_count > 0:
        console.print(f"[dim]Strategy YAMLs: {strategy_dir}[/dim]")
    else:
        console.print("[dim]No strategy YAML artifacts were written.[/dim]")
    if champion_yaml is not None:
        console.print(f"[dim]Champion YAML: {champion_yaml}[/dim]")
    if _trajectory_files:
        console.print(
            f"[dim]Trajectory data: {len(_trajectory_files)} files in {out_path / 'trajectory'}[/dim]"
        )
    if result.early_stopped and not result.rounds:
        raise typer.Exit(1)


def evolve_islands_command(
    strategy_id: str = typer.Argument(..., help="Base strategy ID to evolve"),
    islands: int = typer.Option(3, "--islands", "-n", help="Number of parallel islands"),
    generations: int = typer.Option(3, "--generations", "-g", help="Number of generations"),
    rounds_per_gen: int = typer.Option(
        2, "--rounds-per-gen", help="Evolution rounds per island per generation"
    ),
    samples: int = typer.Option(60, "--samples", "-s", help="Max stocks per backtest"),
    start_date: str | None = typer.Option(None, "--start", help="Start date (YYYY-MM-DD)"),
    end_date: str | None = typer.Option(None, "--end", help="End date (YYYY-MM-DD)"),
    adapter: str | None = typer.Option(None, "--adapter", help="Data adapter override"),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
) -> None:
    """Multi-island parallel evolution for diverse strategy exploration."""
    from alphaevo.orchestrator.islands import IslandEvolution

    dr: tuple[date, date] | None = None
    if start_date:
        s = date.fromisoformat(start_date)
        e = date.fromisoformat(end_date) if end_date else date.today()
        dr = (s, e)

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if model:
        overrides.setdefault("llm", {})["model"] = model

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"🏝️ Island Evolution: [cyan]{strategy_id}[/cyan]\n"
            f"Islands: {islands} | Generations: {generations} | "
            f"Rounds/gen: {rounds_per_gen}",
            style="bold magenta",
        )
    )

    island_evo = IslandEvolution(
        config,
        n_islands=islands,
        rounds_per_generation=rounds_per_gen,
        generations=generations,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting island evolution...", total=None)

        try:
            result = island_evo.evolve(
                [strategy_id],
                max_symbols=samples,
                date_range=dr,
                on_progress=lambda msg: progress.update(task, description=msg),
            )
        except Exception as e:
            logger.exception("Island evolution failed")
            console.print(f"[red]Island evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Summary table
    table = Table(title="🏝️ Island Evolution Results")
    table.add_column("Island", justify="right", style="bold")
    table.add_column("Champion", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Generation")

    for island in result.islands:
        table.add_row(
            str(island.island_id),
            island.champion_id or "—",
            f"{island.champion_score:.1%}",
            str(island.generation),
        )
    console.print(table)

    console.print(
        f"\n🏆 Global champion: [cyan]{result.global_champion_id}[/cyan] "
        f"(score: {result.global_champion_score:.1%})"
    )
    console.print(
        f"📊 Total evaluations: {result.total_evaluations} | "
        f"Migrations: {result.migrations} | "
        f"Diversity: {result.diversity_score:.2f}"
    )


def evolve_curriculum_command(
    strategy_id: str = typer.Argument(..., help="Strategy ID to train progressively"),
    adapter: str | None = typer.Option(None, "--adapter", help="Data adapter override"),
    model: str | None = typer.Option(None, "--model", help="LLM model override"),
) -> None:
    """Progressive difficulty evolution (easy -> medium -> hard -> reality)."""
    from alphaevo.orchestrator.curriculum import CurriculumEvolution

    overrides: dict = {}
    if adapter:
        overrides.setdefault("data", {})["adapter"] = adapter
    if model:
        overrides.setdefault("llm", {})["model"] = model

    config = _get_config(overrides or None)

    console.print(
        Panel(
            f"📚 Curriculum Evolution: [cyan]{strategy_id}[/cyan]\n"
            "Stages: Easy → Medium → Hard → Reality",
            style="bold magenta",
        )
    )

    curriculum = CurriculumEvolution(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Starting curriculum evolution...", total=None)

        try:
            result = curriculum.evolve(
                strategy_id,
                on_progress=lambda msg: progress.update(task, description=msg),
            )
        except Exception as e:
            logger.exception("Curriculum evolution failed")
            console.print(f"[red]Curriculum evolution failed: {e}[/red]")
            console.print("[dim]Set ALPHAEVO_LOG_LEVEL=DEBUG for full traceback[/dim]")
            raise typer.Exit(1) from None

    # Stage results
    table = Table(title="📚 Curriculum Results")
    table.add_column("Stage", style="bold")
    table.add_column("Score", justify="right")
    table.add_column("Status")

    all_stages = ["easy", "medium", "hard", "reality"]
    for stage in all_stages:
        if stage in result.stage_scores:
            score = result.stage_scores[stage]
            status = "[green]✅ Graduated[/green]"
            table.add_row(stage.upper(), f"{score:.1%}", status)
        elif stage in result.stages_completed:
            table.add_row(stage.upper(), "—", "[red]❌ Failed[/red]")
        else:
            table.add_row(stage.upper(), "—", "[dim]⏭️ Skipped[/dim]")
    console.print(table)

    if result.graduated:
        console.print(
            f"\n🎓 Graduated all stages! Champion: [cyan]{result.champion_id}[/cyan] "
            f"(score: {result.champion_score:.1%})"
        )
    else:
        console.print(f"\n📊 Completed {len(result.stages_completed)}/4 stages")
        if result.champion_id:
            console.print(
                f"  Best so far: [cyan]{result.champion_id}[/cyan] "
                f"(score: {result.champion_score:.1%})"
            )
