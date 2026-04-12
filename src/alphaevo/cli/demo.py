"""AlphaEvo demo — shows the full self-evolution loop with synthetic data.

Demonstrates: strategy v1 → backtest → reflect → mutate → v2 → … → champion.
No network access, API keys, or LLM needed — uses heuristic reflection.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.core.config import ConfigManager
from alphaevo.data.adapter import DataManager
from alphaevo.data.cache import DataCache
from alphaevo.evaluator.metrics import Evaluator
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import EvaluationReport, SampleBatch, StrategyChange
from alphaevo.reflection.analyzer import ReflectionAnalyzer
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.reflection.meta_learner import MetaLearner
from alphaevo.reflection.mutator import StrategyMutator
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.library import PatternLibrary
from alphaevo.strategy.tunable import is_period_tunable_target, resolve_tunable_target

if TYPE_CHECKING:
    from rich.console import Console

    from alphaevo.data.adapter import DataAdapter
    from alphaevo.models.strategy import Strategy


def _generate_synthetic_ohlcv(
    symbol: str,
    days: int = 120,
    base_price: float = 50.0,
    phases: list[tuple[int, float, float]] | None = None,
    volume_spikes: list[int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data with configurable trend phases."""
    rng = random.Random(seed)

    if phases is None:
        phases = [(days, 0.001, 0.02)]
    spike_days = set(volume_spikes or [])

    rows: list[dict] = []
    price = base_price
    day = 0
    for num_days, trend, volatility in phases:
        for _ in range(num_days):
            if day >= days:
                break
            daily_return = trend + rng.gauss(0, volatility)
            price *= 1 + daily_return
            price = max(price, 0.5)

            o = price * (1 + rng.uniform(-0.005, 0.005))
            h = max(o, price) * (1 + rng.uniform(0, 0.015))
            low = min(o, price) * (1 - rng.uniform(0, 0.015))
            c = price
            base_vol = rng.uniform(800_000, 2_000_000)
            vol = int(base_vol * (rng.uniform(2.0, 3.5) if day in spike_days else 1.0))
            prev_c = rows[-1]["close"] if rows else c * 0.99

            rows.append(
                {
                    "date": date(2024, 6, 1) + timedelta(days=day),
                    "open": round(o, 2),
                    "high": round(h, 2),
                    "low": round(low, 2),
                    "close": round(c, 2),
                    "volume": vol,
                    "prev_close": round(prev_c, 2),
                }
            )
            day += 1
    return pd.DataFrame(rows)


def _find_builtin_strategy_dir() -> Path | None:
    """Locate builtin strategies directory."""
    candidates = [
        Path(__file__).resolve().parent.parent.parent.parent / "strategies" / "builtin",
        Path("strategies/builtin"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


class _DummyLLM:
    """No-op LLM that forces heuristic fallback in demos."""

    def chat(self, *a: object, **kw: object) -> str:
        raise RuntimeError("demo")

    def reflect_json(self, *a: object, **kw: object) -> dict:
        raise RuntimeError("demo")


@dataclass
class _DemoMutationCandidate:
    """A validated candidate mutation for showcase-style demos."""

    strategy: Strategy
    evaluation: EvaluationReport
    signals: int
    trades: list
    changes: list[StrategyChange]


def _load_demo_strategy(parser: StrategyParser, builtin_dir: Path) -> Strategy:
    """Load the most showcase-friendly builtin strategy for the synthetic demo."""
    preferred = [
        "ma_crossover.yaml",
        "trend_pullback_rebound.yaml",
    ]
    for filename in preferred:
        strategy_file = builtin_dir / filename
        if strategy_file.exists():
            return parser.parse_file(strategy_file)

    yaml_files = list(builtin_dir.glob("*.yaml"))
    if not yaml_files:
        raise FileNotFoundError("No strategy files found")
    return parser.parse_file(yaml_files[0])


def _infer_demo_change_type(
    target: str,
    current: float | int,
    candidate: float | int,
) -> ChangeType:
    """Infer a readable change type for demo-generated tunable candidates."""
    if target.startswith("exit."):
        return ChangeType.ADJUST_EXIT

    if is_period_tunable_target(target):
        if target.endswith(".indicator.fast"):
            return ChangeType.TIGHTEN_FILTER if candidate > current else ChangeType.LOOSEN_FILTER
        if target.endswith(".indicator.slow") or target.endswith(".indicator.signal"):
            return ChangeType.TIGHTEN_FILTER if candidate < current else ChangeType.LOOSEN_FILTER
        if target.endswith(".indicator.std"):
            return ChangeType.TIGHTEN_FILTER if candidate > current else ChangeType.LOOSEN_FILTER
        return ChangeType.TIGHTEN_FILTER if candidate < current else ChangeType.LOOSEN_FILTER

    if "stop_loss" in target or "take_profit" in target:
        return ChangeType.ADJUST_EXIT
    return ChangeType.TIGHTEN_FILTER if candidate > current else ChangeType.LOOSEN_FILTER


def _build_neighbor_change_candidates(strategy: Strategy) -> list[list[StrategyChange]]:
    """Generate small one-step tunable mutations around the current strategy."""
    if strategy.params is None or not strategy.params.tunable:
        return []

    candidates: list[list[StrategyChange]] = []
    for param in strategy.params.tunable:
        current = resolve_tunable_target(strategy, param.target)
        if not isinstance(current, (int, float)):
            continue

        lo, hi = param.range
        step = param.step or (hi - lo) / 10
        label = param.label or param.target

        for raw_candidate in (current - step, current + step):
            candidate: int | float
            if is_period_tunable_target(param.target):
                candidate = int(round(raw_candidate))
            else:
                candidate = round(float(raw_candidate), 4)

            if candidate == current or candidate < lo or candidate > hi:
                continue

            change = StrategyChange(
                change_type=_infer_demo_change_type(param.target, current, candidate),
                target=param.target,
                from_value=current,
                to_value=candidate,
                reason=f"Validated {label} from {current} to {candidate} on the demo batch",
            )
            candidates.append([change])

    return candidates


def _candidate_signature(changes: list[StrategyChange]) -> tuple[tuple[str, str, str], ...]:
    """Build a stable signature for deduplicating candidate mutations."""
    signature = [
        (change.change_type.value, change.target, str(change.to_value)) for change in changes
    ]
    return tuple(sorted(signature))


def _select_best_demo_mutation(
    strategy: Strategy,
    evaluation: EvaluationReport,
    data: dict[str, pd.DataFrame],
    *,
    analyzer: ReflectionAnalyzer,
    critic: SelfCritic,
    mutator: StrategyMutator,
) -> _DemoMutationCandidate | None:
    """Test candidate mutations and keep only a score-improving one."""
    reflection = analyzer.reflect(strategy, evaluation)
    verdict = critic.critique(strategy, evaluation, reflection)

    candidate_change_sets: list[list[StrategyChange]] = []
    if verdict.approved:
        candidate_change_sets.append(verdict.approved)
        candidate_change_sets.extend([[change] for change in verdict.approved])
    candidate_change_sets.extend(_build_neighbor_change_candidates(strategy))

    baseline_score = evaluation.confidence_score
    best: _DemoMutationCandidate | None = None
    seen: set[tuple[tuple[str, str, str], ...]] = set()

    for changes in candidate_change_sets:
        if not changes:
            continue
        signature = _candidate_signature(changes)
        if signature in seen:
            continue
        seen.add(signature)

        try:
            candidate_strategy = mutator.mutate(strategy, changes, atomic=True)
        except Exception:
            continue

        candidate_eval, candidate_signals, candidate_trades = _run_backtest(
            candidate_strategy, data
        )
        if candidate_eval.confidence_score <= baseline_score:
            continue

        if best is None or candidate_eval.confidence_score > best.evaluation.confidence_score:
            best = _DemoMutationCandidate(
                strategy=candidate_strategy,
                evaluation=candidate_eval,
                signals=candidate_signals,
                trades=candidate_trades,
                changes=changes,
            )

    return best


def _record_demo_experience(
    experience_store: ExperienceStore,
    *,
    strategy_family: str,
    strategy_id: str,
    round_num: int,
    changes: list[StrategyChange],
    score_before: float,
    score_after: float,
) -> None:
    """Record only validated demo outcomes, using real score deltas."""
    if not changes:
        return

    score_delta = round(score_after - score_before, 4)
    worked = score_delta > 0
    experience_store.record_batch(
        [
            ExperienceRecord(
                strategy_family=strategy_family,
                strategy_id=strategy_id,
                round_num=round_num,
                change_type=change.change_type,
                target=change.target,
                from_value=change.from_value,
                to_value=change.to_value,
                reason=change.reason,
                score_before=score_before,
                score_after=score_after,
                score_delta=score_delta,
                worked=worked,
                lesson=change.reason,
                action_type="demo_validated_search",
                source="demo",
            )
            for change in changes
        ]
    )


def _build_synthetic_data() -> dict[str, pd.DataFrame]:
    """Build 12 synthetic stocks with diverse patterns.

    NOTE: This data is synthetic and designed to produce a variety of
    market conditions (uptrend, downtrend, choppy, V-recovery).
    Results are illustrative of the evolution workflow, not predictive.
    """
    # Balanced mix: ~3 favorable, ~4 neutral/mixed, ~5 adverse/traps.
    # This ensures the demo doesn't over-promise strategy performance.
    stock_configs: list[tuple[str, float, list[tuple[int, float, float]], list[int]]] = [
        # ── Favorable (3/12) ── uptrend with pullback → bounce
        (
            "DEMO001",
            50.0,
            [
                (45, 0.005, 0.008),
                (3, -0.005, 0.005),
                (4, 0.004, 0.012),
                (30, 0.003, 0.008),
                (3, -0.004, 0.005),
                (4, 0.003, 0.010),
                (31, 0.002, 0.010),
            ],
            [48, 49, 78, 79],
        ),
        (
            "DEMO002",
            75.0,
            [
                (40, 0.006, 0.007),
                (3, -0.006, 0.004),
                (5, 0.005, 0.010),
                (35, 0.004, 0.007),
                (3, -0.005, 0.004),
                (5, 0.004, 0.010),
                (29, 0.002, 0.010),
            ],
            [43, 44, 83, 84],
        ),
        (
            "DEMO003",
            100.0,
            [
                (50, 0.004, 0.009),
                (4, -0.004, 0.005),
                (3, 0.005, 0.012),
                (25, 0.003, 0.009),
                (4, -0.003, 0.005),
                (3, 0.004, 0.010),
                (31, 0.002, 0.010),
            ],
            [54, 55, 86, 87],
        ),
        # ── Neutral/Mixed (4/12) ── some signals but outcome uncertain
        (
            "DEMO004",
            60.0,
            [
                (60, 0.002, 0.012),  # slow uptrend with noise
                (15, -0.002, 0.010),  # modest pullback
                (45, 0.001, 0.013),  # sideways drift
            ],
            [75, 76],
        ),
        (
            "DEMO005",
            85.0,
            [
                (30, 0.004, 0.010),  # up
                (20, -0.001, 0.014),  # sideways with large noise
                (30, 0.003, 0.012),  # resume up but noisy
                (40, -0.001, 0.011),  # fade out
            ],
            [50, 80],
        ),
        (
            "DEMO006",
            110.0,
            [(50, 0.003, 0.012), (8, -0.003, 0.008), (62, 0.002, 0.012)],
            [58, 59, 90],
        ),
        (
            "DEMO007",
            40.0,
            [
                (40, 0.003, 0.011),  # mild up
                (10, -0.006, 0.009),  # sharp correction
                (35, 0.002, 0.013),  # recovery with high vol
                (35, 0.001, 0.012),  # drift
            ],
            [50, 75],
        ),
        # ── Adverse (5/12) ── downtrend, choppy, traps, false signals
        ("DEMO008", 90.0, [(120, -0.003, 0.018)], []),  # pure bear
        ("DEMO009", 65.0, [(120, 0.0, 0.015)], [40, 80]),  # flat choppy
        (
            "DEMO010",
            80.0,
            [
                (30, -0.005, 0.015),  # bear phase
                (10, -0.002, 0.010),  # bottoming with false signals
                (25, 0.006, 0.008),  # recovery rally
                (3, -0.004, 0.005),  # pullback
                (4, 0.005, 0.010),  # bounce
                (48, 0.003, 0.010),  # uptrend
            ],
            [40, 41, 68, 69],  # signals in bottoming (traps) AND recovery
        ),
        (
            "DEMO011",
            120.0,
            [
                (20, 0.005, 0.008),   # initial up
                (25, -0.004, 0.014),  # crash
                (15, 0.002, 0.012),   # volatile recovery
                (20, -0.003, 0.013),  # second leg down (double bottom trap)
                (40, 0.003, 0.010),   # real recovery
            ],
            [45, 46, 63, 64],  # signals during false AND real recovery
        ),
        (
            "DEMO012",
            70.0,
            [
                (40, 0.004, 0.009),   # uptrend
                (15, -0.006, 0.012),  # sharp selloff
                (5, 0.008, 0.015),    # V-bounce (false signal)
                (10, -0.003, 0.010),  # fails — double top trap
                (15, -0.004, 0.012),  # extended decline
                (3, -0.004, 0.005),   # washout
                (4, 0.005, 0.010),    # real bounce
                (28, 0.003, 0.010),   # recovery
            ],
            [55, 56, 60, 90, 91],  # false AND real signals mixed
        ),
    ]

    data: dict[str, pd.DataFrame] = {}
    for idx, (sym, base, phases, spikes) in enumerate(stock_configs):
        data[sym] = _generate_synthetic_ohlcv(
            sym,
            days=120,
            base_price=base,
            phases=phases,
            volume_spikes=spikes,
            seed=42 + idx,
        )
    return data


def _run_backtest(
    strategy: Strategy,
    data: dict[str, pd.DataFrame],
) -> tuple[EvaluationReport, int, list]:
    """Run a single backtest + evaluation, return (report, signal_count, signals)."""
    batch = SampleBatch(
        batch_id="demo_batch",
        strategy_id=strategy.meta.id,
        symbols=list(data.keys()),
        date_range=(date(2024, 6, 1), date(2024, 9, 28)),
    )
    engine = BacktestEngine(slippage=0.001, commission=0.0003, min_data_days=30)
    bt_result = engine.run(strategy, data, batch)
    evaluator = Evaluator()
    report = evaluator.evaluate(bt_result, strategy)
    return report, bt_result.total_signals, bt_result.signals


def _display_round(
    console: Console,
    round_num: int,
    strategy: Strategy,
    evaluation: EvaluationReport,
    signals: int,
    prev_score: float | None = None,
) -> None:
    """Display a single round's results compactly."""
    m = evaluation.overall
    score = evaluation.confidence_score
    assessment = strategy.assess_market_hypothesis(evaluation)

    delta = ""
    if prev_score is not None:
        diff = score - prev_score
        if diff > 0:
            delta = f"  [bold green]↑ +{diff:.1%}[/bold green]"
        elif diff < 0:
            delta = f"  [bold red]↓ {diff:.1%}[/bold red]"
        else:
            delta = "  [dim]→ same[/dim]"

    console.print(
        f"  [bold]Round {round_num}[/bold] │ "
        f"[cyan]{strategy.meta.id}[/cyan] │ "
        f"Win: {m.win_rate:.0%}  P/L: {m.profit_loss_ratio:.1f}  "
        f"DD: {m.max_drawdown:.0%}  Signals: {signals}  │ "
        f"Score: [bold]{score:.1%}[/bold]{delta}"
    )
    console.print(
        "    🧠 [bold]Hypothesis lens:[/bold] "
        f"{assessment.status.replace('_', ' ')} — {assessment.rationale}"
    )
    console.print(f"    ↳ Next step: {assessment.next_step}")
    if signals < 30:
        console.print(
            "    🛑 [yellow]Sample adequacy:[/yellow] fewer than 30 signals on this batch; "
            "treat any improvement as provisional."
        )
    elif evaluation.anti_overfit.is_overfit:
        console.print(
            "    🛑 [yellow]Overfit gate:[/yellow] this version would not be promoted as "
            "champion without better generalization."
        )


def _display_changes(
    console: Console,
    changes: list[StrategyChange],
) -> None:
    """Show what mutations were applied."""
    for ch in changes:
        icon = {
            ChangeType.TIGHTEN_FILTER: "🔧",
            ChangeType.LOOSEN_FILTER: "🔓",
            ChangeType.ADJUST_EXIT: "🎯",
            ChangeType.ADD_CONDITION: "➕",
            ChangeType.REMOVE_CONDITION: "➖",
            ChangeType.CHANGE_UNIVERSE: "🌐",
            ChangeType.CHANGE_LOGIC: "🔀",
            ChangeType.DISCOVER_FACTOR: "🧪",
        }.get(ch.change_type, "•")
        console.print(f"    {icon} [dim]{ch.change_type.value}[/dim]: {ch.reason}")


def _persist_demo_history(
    history: list[tuple[Strategy, EvaluationReport, int, list]],
) -> int:
    """Persist demo-generated strategies and evaluations to the configured store."""
    if not history:
        return 0

    from alphaevo.strategy.store import StrategyStore

    config = ConfigManager().load()
    store = StrategyStore(config.db_path)
    saved = 0
    for strategy, evaluation, _signals, _trades in history:
        store.save(strategy)
        store.save_evaluation(evaluation)
        saved += 1
    return saved


def run_demo(console: Console) -> None:
    """Execute a complete self-evolution demo with synthetic data."""
    console.print(
        Panel(
            "[bold cyan]🧬 AlphaEvo — Self-Evolution Demo[/bold cyan]\n\n"
            "Watch a strategy improve through validated rounds of\n"
            "backtest → reflect → test candidate mutations → keep the best one.\n\n"
            "[dim]Uses synthetic data — results illustrate the workflow only.\n"
            "Run [cyan]alphaevo demo --real[/cyan] for real market data.[/dim]",
            style="bold",
        )
    )

    # ── Load strategy ──────────────────────────────────────────────────
    builtin_dir = _find_builtin_strategy_dir()
    if builtin_dir is None:
        console.print("[red]Could not find strategies/builtin/ directory[/red]")
        return

    parser = StrategyParser()
    try:
        strategy = _load_demo_strategy(parser, builtin_dir)
    except FileNotFoundError:
        console.print("[red]No strategy files found[/red]")
        return

    # ── Generate data ──────────────────────────────────────────────────
    data = _build_synthetic_data()
    console.print(
        f"  [green]✓[/green] Loaded [cyan]{strategy.meta.name}[/cyan] "
        f"│ {len(data)} synthetic stocks × 120 days"
    )

    # ── Evolution loop ─────────────────────────────────────────────────
    max_rounds = 4
    console.print(
        f"\n[bold]{'═' * 60}[/bold]"
        f"\n[bold]  🔬 Evolution: up to {max_rounds} validated rounds[/bold]"
        f"\n[bold]{'═' * 60}[/bold]"
    )

    mutator = StrategyMutator(max_changes=3, complexity_limit=8)

    # Self-evolution modules (in-memory for demo)
    pattern_library = PatternLibrary(db_path=":memory:")
    experience_store = ExperienceStore(db_path=":memory:")
    meta_learner = MetaLearner(experience_store)
    critic = SelfCritic(experience_store=experience_store, complexity_limit=8)

    analyzer = ReflectionAnalyzer(llm=_DummyLLM(), max_changes=3)  # type: ignore[arg-type]

    current = strategy
    champion = strategy
    champion_score = 0.0
    history: list[tuple[Strategy, EvaluationReport, int, list]] = []

    # Suppress reflection fallback warning in demo
    import logging

    logging.getLogger("alphaevo.reflection.analyzer").setLevel(logging.ERROR)

    for round_num in range(1, max_rounds + 1):
        console.print()

        # Backtest
        evaluation, signals, trades = _run_backtest(current, data)
        score = evaluation.confidence_score
        prev = history[-1][1].confidence_score if history else None
        _display_round(console, round_num, current, evaluation, signals, prev)

        history.append((current, evaluation, signals, trades))

        if score > champion_score:
            champion = current
            champion_score = score
            # Extract patterns from champion
            try:
                patterns = pattern_library.extract_patterns_from_strategy(current, evaluation)
                for p in patterns:
                    pattern_library.save(p)
                if patterns:
                    console.print(f"    [dim]📚 Extracted {len(patterns)} reusable patterns[/dim]")
            except Exception:
                pass

        # Reflect, validate, and adopt only score-improving changes (skip on last round)
        if round_num < max_rounds:
            candidate = _select_best_demo_mutation(
                current,
                evaluation,
                data,
                analyzer=analyzer,
                critic=critic,
                mutator=mutator,
            )
            if candidate is None:
                console.print("    [dim]No validated improvement found — champion locked in[/dim]")
                break
            _display_changes(console, candidate.changes)
            console.print(
                "    [green]✅ validated on the demo batch:[/green] "
                f"{score:.1%} → {candidate.evaluation.confidence_score:.1%}"
            )
            _record_demo_experience(
                experience_store,
                strategy_family="demo",
                strategy_id=current.meta.id,
                round_num=round_num,
                changes=candidate.changes,
                score_before=score,
                score_after=candidate.evaluation.confidence_score,
            )
            current = candidate.strategy

    # ── Evolution summary ──────────────────────────────────────────────
    console.print(f"\n[bold]{'═' * 60}[/bold]")

    table = Table(
        title="🧬 Evolution Summary",
        show_lines=True,
    )
    table.add_column("Version", style="cyan", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("P/L Ratio", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Score", justify="right")

    for strat, ev, sigs, _trades in history:
        m = ev.overall
        is_champ = strat.meta.id == champion.meta.id
        score_str = (
            f"[bold green]{ev.confidence_score:.1%}[/bold green]"
            if is_champ
            else f"{ev.confidence_score:.1%}"
        )
        version_str = f"[bold]{strat.meta.id}[/bold] 🏆" if is_champ else strat.meta.id
        table.add_row(
            version_str,
            f"{m.win_rate:.1%}",
            f"{m.profit_loss_ratio:.2f}",
            f"{m.max_drawdown:.1%}",
            f"{m.sharpe_ratio:.2f}",
            str(sigs),
            score_str,
        )
    console.print(table)

    # Evolution progress chart
    round_scores = [(strat.meta.id, ev.confidence_score) for strat, ev, _, _ in history]
    from alphaevo.evaluator.reporter import Reporter

    evo_chart = Reporter.plot_evolution_scores(round_scores, title="🧬 Evolution Progress")
    console.print(Text.from_ansi(f"\n{evo_chart}"))

    # Improvement summary
    if len(history) >= 2:
        first_score = history[0][1].confidence_score
        improvement = champion_score - first_score
        if improvement > 0:
            console.print(
                f"\n  📈 Strategy improved from "
                f"[red]{first_score:.1%}[/red] → "
                f"[bold green]{champion_score:.1%}[/bold green] "
                f"([bold green]+{improvement:.1%}[/bold green])"
            )
        console.print(f"  🏆 Champion: [bold cyan]{champion.meta.id}[/bold cyan]")

    # Show sample trades from champion
    champ_entry = [h for h in history if h[0].meta.id == champion.meta.id][0]
    champ_trades = champ_entry[3]
    if champ_trades:
        console.print()
        # Sort: show best winners and worst losers for a balanced view
        sorted_trades = sorted(champ_trades, key=lambda s: s.return_pct, reverse=True)
        top_winners = sorted_trades[:3]
        top_losers = sorted_trades[-2:] if len(sorted_trades) > 3 else []
        sample = top_winners + top_losers

        sig_table = Table(title="📋 Sample Trades (champion — top wins & losses)")
        sig_table.add_column("Symbol", style="cyan")
        sig_table.add_column("Date")
        sig_table.add_column("Entry", justify="right")
        sig_table.add_column("Exit", justify="right")
        sig_table.add_column("Return", justify="right")
        sig_table.add_column("Reason")

        for s in sample:
            ret_style = "green" if s.return_pct > 0 else "red"
            sig_table.add_row(
                s.symbol,
                str(s.signal_date),
                f"{s.entry_price:.2f}",
                f"{s.exit_price:.2f}" if s.exit_price else "—",
                f"[{ret_style}]{s.return_pct:.2%}[/{ret_style}]",
                s.exit_reason.value if s.exit_reason else "—",
            )
        console.print(sig_table)

        # Equity curve for champion
        eq_chart = Reporter.plot_equity_curve(
            champ_trades, title=f"📈 Champion Equity — {champion.meta.id}"
        )
        console.print(Text.from_ansi(f"\n{eq_chart}"))

        dist_chart = Reporter.plot_return_distribution(
            champ_trades, title="📊 Return Distribution"
        )
        console.print(Text.from_ansi(f"\n{dist_chart}"))

    # Meta-learning summary
    profile = meta_learner.analyze(family_id="demo")
    meta_insights = [i for i in profile.insights if i.confidence > 0.1]
    if meta_insights:
        console.print("\n  [bold]🧠 Meta-Insights from Evolution:[/bold]")
        for ins in meta_insights:
            console.print(f"    • {ins.description}")

    # Pattern library summary
    all_patterns = pattern_library.get_best_patterns(min_score=0.0)
    if all_patterns:
        console.print(
            f"  [bold]📚 Pattern Library:[/bold] {len(all_patterns)} patterns accumulated"
        )

    try:
        saved = _persist_demo_history(history)
    except Exception:
        saved = 0
    if saved:
        console.print(f"  💾 Saved {saved} demo strategy snapshots to the local strategy store")
        console.print(f"  🔎 Inspect with: [cyan]alphaevo strategy show {champion.meta.id}[/cyan]")

    # Next steps
    console.print(
        Panel(
            "[bold green]✨ Demo complete![/bold green]\n\n"
            "[yellow]⚠️  Results above use synthetic data and are illustrative only.[/yellow]\n\n"
            "Try it yourself:\n"
            "  [cyan]alphaevo demo --real[/cyan]                — demo with real market data\n"
            "  [cyan]alphaevo run <strategy_id>[/cyan]           — backtest with real data\n"
            "  [cyan]alphaevo evolve <id> --rounds 5[/cyan]      — self-evolve (with LLM)\n"
            "  [cyan]alphaevo strategy create[/cyan]             — create from description\n"
            "  [cyan]alphaevo leaderboard[/cyan]                 — view rankings\n\n"
            "[dim]⚠️ Research tool only — not investment advice.[/dim]",
            style="bold",
        )
    )


def run_real_demo(console: Console, *, market: str = "us") -> None:
    """Run a demo using real market data from yfinance (US) or akshare (CN).

    Requires network access. No API key or LLM needed.
    """
    import asyncio

    console.print(
        Panel(
            "[bold cyan]🧬 AlphaEvo — Real Data Demo[/bold cyan]\n\n"
            f"Market: [bold]{market.upper()}[/bold]\n"
            "Uses heuristic reflection (no LLM needed).\n"
            "May stop early when real data does not justify another mutation.\n\n"
            "[dim]Requires network access to download market data.[/dim]",
            style="bold",
        )
    )

    # Pick strategy + symbols based on market
    builtin_dir = _find_builtin_strategy_dir()
    if builtin_dir is None:
        console.print("[red]Could not find strategies/builtin/ directory[/red]")
        return

    if market == "cn":
        strategy_file = builtin_dir / "mean_reversion_oversold.yaml"
        symbols = ["600519", "000858", "601318", "000333", "600036"]
        adapter_name = "akshare"
    else:
        strategy_file = builtin_dir / "ma_crossover.yaml"
        if not strategy_file.exists():
            strategy_file = builtin_dir / "rsi_reversion.yaml"
        symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        adapter_name = "yfinance"

    if not strategy_file.exists():
        yaml_files = list(builtin_dir.glob("*.yaml"))
        if not yaml_files:
            console.print("[red]No strategy files found[/red]")
            return
        strategy_file = yaml_files[0]

    parser = StrategyParser()
    strategy = parser.parse_file(strategy_file)

    console.print(
        f"  [green]✓[/green] Strategy: [cyan]{strategy.meta.name}[/cyan]\n"
        f"  [green]✓[/green] Symbols: {', '.join(symbols)}\n"
        f"  [green]✓[/green] Adapter: {adapter_name}"
    )

    # Download data
    console.print("\n  Downloading market data...")
    try:
        data = asyncio.run(_fetch_real_data(symbols, adapter_name))
    except Exception as e:
        console.print(f"[red]Data download failed: {e}[/red]")
        console.print("[dim]Try: alphaevo demo  (uses synthetic data, no network needed)[/dim]")
        return

    if not data:
        console.print("[red]No data received. Check your network connection.[/red]")
        return

    console.print(
        f"  [green]✓[/green] Downloaded {len(data)} symbols, "
        f"~{sum(len(df) for df in data.values())} total rows"
    )

    # Run backtest + evolution using the same demo loop
    console.print(
        f"\n[bold]{'═' * 60}[/bold]"
        f"\n[bold]  🔬 Real Data Evolution: up to 3 validated rounds[/bold]"
        f"\n[bold]{'═' * 60}[/bold]"
    )

    mutator = StrategyMutator(max_changes=3, complexity_limit=8)
    experience_store = ExperienceStore(db_path=":memory:")
    critic = SelfCritic(experience_store=experience_store, complexity_limit=8)

    analyzer = ReflectionAnalyzer(llm=_DummyLLM(), max_changes=3)  # type: ignore[arg-type]

    import logging

    logging.getLogger("alphaevo.reflection.analyzer").setLevel(logging.ERROR)

    current = strategy
    champion = strategy
    champion_score = 0.0
    history: list[tuple[Strategy, EvaluationReport, int, list]] = []

    for round_num in range(1, 4):
        console.print()
        evaluation, signals, trades = _run_backtest(current, data)
        score = evaluation.confidence_score
        prev = history[-1][1].confidence_score if history else None
        _display_round(console, round_num, current, evaluation, signals, prev)
        history.append((current, evaluation, signals, trades))

        if score > champion_score:
            champion = current
            champion_score = score

        if round_num < 3:
            candidate = _select_best_demo_mutation(
                current,
                evaluation,
                data,
                analyzer=analyzer,
                critic=critic,
                mutator=mutator,
            )
            if candidate is None:
                console.print(
                    "    [yellow]⏹️ No validated improvement found on real data — stopping.[/yellow]"
                )
                console.print(
                    "    [dim]This is an intended research outcome: AlphaEvo prefers an "
                    "explicit early stop over forcing a prettier score.[/dim]"
                )
                break
            _display_changes(console, candidate.changes)
            console.print(
                "    [green]✅ validated on the downloaded batch:[/green] "
                f"{score:.1%} → {candidate.evaluation.confidence_score:.1%}"
            )
            current = candidate.strategy

    # Summary
    console.print(f"\n[bold]{'═' * 60}[/bold]")
    table = Table(title="🧬 Real Data Evolution Summary", show_lines=True)
    table.add_column("Version", style="cyan", justify="center")
    table.add_column("Win Rate", justify="right")
    table.add_column("P/L Ratio", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Signals", justify="right")
    table.add_column("Score", justify="right")
    for strat, ev, sigs, _ in history:
        is_champ = strat.meta.id == champion.meta.id
        score_str = (
            f"[bold green]{ev.confidence_score:.1%}[/bold green]"
            if is_champ
            else f"{ev.confidence_score:.1%}"
        )
        table.add_row(
            f"[bold]{strat.meta.id}[/bold] 🏆" if is_champ else strat.meta.id,
            f"{ev.overall.win_rate:.1%}",
            f"{ev.overall.profit_loss_ratio:.2f}",
            f"{ev.overall.max_drawdown:.1%}",
            str(sigs),
            score_str,
        )
    console.print(table)
    if history:
        first_score = history[0][1].confidence_score
        improvement = champion_score - first_score
        if improvement > 0:
            console.print(
                f"  📈 Real-data champion improvement: {first_score:.1%} → "
                f"[bold green]{champion_score:.1%}[/bold green] "
                f"([bold green]+{improvement:.1%}[/bold green])"
            )
        else:
            console.print(
                "  ⏹️ Real-data demo ended without a validated gain. "
                "That is still a useful research result."
            )
    try:
        saved = _persist_demo_history(history)
    except Exception:
        saved = 0
    if saved:
        console.print(f"  💾 Saved {saved} real-data demo snapshots to the local strategy store")
    console.print(
        Panel(
            "[bold green]✨ Real data demo complete![/bold green]\n\n"
            "[dim]⚠️ Research tool only — not investment advice.[/dim]",
            style="bold",
        )
    )


async def _fetch_real_data(symbols: list[str], adapter_name: str) -> dict[str, pd.DataFrame]:
    """Fetch real OHLCV data using the specified adapter."""
    end = date.today()
    start = end - timedelta(days=180)

    if adapter_name == "akshare":
        from alphaevo.data.adapters.akshare import AkShareAdapter

        adapter: DataAdapter = AkShareAdapter()
    else:
        from alphaevo.data.adapters.yfinance import YFinanceAdapter

        adapter = YFinanceAdapter()

    cache_dir = ConfigManager().load().data.cache_dir
    data_manager = DataManager([adapter], cache=DataCache(cache_dir))

    async def _fetch_one(sym: str) -> tuple[str, pd.DataFrame | None]:
        try:
            df = await data_manager.get_history(sym, start, end)
            if not df.empty and len(df) >= 30:
                return (sym, df)
        except Exception:
            pass  # Skip symbols that fail
        return (sym, None)

    results = await asyncio.gather(*[_fetch_one(sym) for sym in symbols])
    return {sym: df for sym, df in results if df is not None}
