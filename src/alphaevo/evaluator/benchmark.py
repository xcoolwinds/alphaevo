"""Benchmark comparator — strategy vs buy-and-hold baselines.

Provides:
1. Buy-and-hold benchmark (equal-weight across symbols)
2. Random signal baseline (with confidence interval)
3. Comparison summary table
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from statistics import mean, stdev

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from alphaevo.models.execution import (
    BenchmarkComparison,
    StressWindowCase,
    StressWindowMetrics,
    TradeSignal,
)

logger = logging.getLogger(__name__)


def _finite_float(value: object) -> float | None:
    """Coerce a scalar to a finite float, returning None for NaN/inf."""
    try:
        scalar = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if scalar.ndim != 0:
        return None
    parsed = float(scalar.item())
    if not np.isfinite(parsed):
        return None
    return parsed


@dataclass
class RandomBaseline:
    """Results from random signal Monte Carlo simulation."""

    mean_return: float = 0.0
    std_return: float = 0.0
    ci_lower: float = 0.0  # 5th percentile
    ci_upper: float = 0.0  # 95th percentile
    beat_fraction: float = 0.0  # fraction of randoms strategy beats
    n_simulations: int = 100


@dataclass
class BenchmarkResult:
    """Full benchmark comparison outcome."""

    buy_hold: BenchmarkComparison
    random_baseline: RandomBaseline | None = None
    stress_windows: StressWindowMetrics | None = None
    summary: str = ""


class BenchmarkComparator:
    """Compare strategy performance against baselines."""

    def __init__(
        self,
        n_random_simulations: int = 100,
        seed: int = 42,
        *,
        stress_window_days: int = 20,
        stress_window_top_k: int = 3,
    ) -> None:
        self._n_sims = n_random_simulations
        self._seed = seed
        self._stress_window_days = stress_window_days
        self._stress_window_top_k = stress_window_top_k

    def compare(
        self,
        trades: list[TradeSignal],
        market_data: dict[str, pd.DataFrame],
        benchmark_df: pd.DataFrame | None = None,
    ) -> BenchmarkResult:
        """Run all benchmark comparisons.

        Args:
            trades: Executed trade signals from strategy backtest.
            market_data: OHLCV data per symbol used in backtest.
        """
        # Buy-and-hold
        bh = self._buy_and_hold(trades, market_data)

        # Random baseline
        random_bl = self._random_baseline(trades, market_data)
        stress_windows = self._stress_window_benchmark(trades, benchmark_df)

        # Summary
        lines = [
            f"Strategy return: {bh.strategy_return:.2%}",
            f"Buy & Hold:      {bh.benchmark_return:.2%}",
            f"Excess (alpha):  {bh.excess_return:+.2%}",
        ]
        if random_bl:
            lines.append(
                f"Random baseline: {random_bl.mean_return:.2%} "
                f"(95% CI: [{random_bl.ci_lower:.2%}, {random_bl.ci_upper:.2%}])"
            )
            lines.append(f"Beats {random_bl.beat_fraction:.0%} of random strategies")
        if stress_windows is not None:
            lines.append(
                f"Stress windows: alpha {stress_windows.average_alpha:+.2%} on average "
                f"(worst {stress_windows.worst_alpha:+.2%})"
            )

        return BenchmarkResult(
            buy_hold=bh,
            random_baseline=random_bl,
            stress_windows=stress_windows,
            summary="\n".join(lines),
        )

    @staticmethod
    def _buy_and_hold(
        trades: list[TradeSignal],
        market_data: dict[str, pd.DataFrame],
    ) -> BenchmarkComparison:
        """Compute buy-and-hold benchmark over the same period."""
        if not market_data:
            return BenchmarkComparison()

        symbol_returns = []
        for _sym, df in market_data.items():
            if df.empty or len(df) < 2:
                continue
            first_close = _finite_float(df["close"].iloc[0])
            last_close = _finite_float(df["close"].iloc[-1])
            if first_close is not None and last_close is not None and first_close > 0:
                symbol_returns.append((last_close - first_close) / first_close)

        if not symbol_returns:
            return BenchmarkComparison()

        bh_return = mean(symbol_returns)

        # Strategy return from trades
        executed = [t for t in trades if t.exit_price is not None]
        if executed:
            executed_returns = [
                ret for trade in executed if (ret := _finite_float(trade.return_pct)) is not None
            ]
            strat_return = (
                mean(executed_returns) * len(executed_returns) / max(len(market_data), 1)
                if executed_returns
                else 0.0
            )
        else:
            strat_return = 0.0

        # Benchmark drawdown (simplified)
        all_closes = []
        for df in market_data.values():
            if df.empty:
                continue
            closes = pd.to_numeric(df["close"], errors="coerce")
            first_close = _finite_float(closes.iloc[0])
            if first_close is None or first_close <= 0:
                continue
            normed = (closes / first_close).replace([np.inf, -np.inf], np.nan).dropna()
            if not normed.empty:
                all_closes.append(normed)

        bh_drawdown = 0.0
        if all_closes:
            avg_curve = pd.concat(all_closes, axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan)
            avg_curve = avg_curve.dropna()
            if not avg_curve.empty:
                peak = avg_curve.cummax()
                drawdowns = ((avg_curve - peak) / peak).replace([np.inf, -np.inf], np.nan).dropna()
                if not drawdowns.empty:
                    bh_drawdown = abs(float(drawdowns.min()))

        # Benchmark Sharpe (simplified daily returns)
        bh_sharpe = 0.0
        if all_closes:
            avg_curve = pd.concat(all_closes, axis=1).mean(axis=1).replace([np.inf, -np.inf], np.nan)
            avg_curve = avg_curve.dropna()
            daily_ret = avg_curve.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
            if len(daily_ret) > 5 and daily_ret.std() > 0:
                bh_sharpe = float(daily_ret.mean() / daily_ret.std() * (252**0.5))

        return BenchmarkComparison(
            benchmark_return=bh_return,
            strategy_return=strat_return,
            excess_return=strat_return - bh_return,
            benchmark_max_drawdown=bh_drawdown,
            benchmark_sharpe=bh_sharpe,
            symbols_used=len(market_data),
        )

    def _random_baseline(
        self,
        trades: list[TradeSignal],
        market_data: dict[str, pd.DataFrame],
    ) -> RandomBaseline | None:
        """Monte Carlo random signal baseline."""
        executed = [t for t in trades if t.exit_price is not None]
        if not executed or not market_data:
            return None

        # Compute the average holding period from real trades
        avg_hold = max(1, int(mean(t.holding_days for t in executed)))
        n_signals = len(executed)

        rng = np.random.default_rng(self._seed)

        # Build pool of possible returns from all symbols
        return_pool: list[float] = []
        for df in market_data.values():
            if len(df) < avg_hold + 1:
                continue
            closes = pd.to_numeric(df["close"], errors="coerce").to_numpy(dtype=float)
            for i in range(len(closes) - avg_hold):
                entry_close = _finite_float(closes[i])
                exit_close = _finite_float(closes[i + avg_hold])
                if entry_close is None or exit_close is None or entry_close <= 0:
                    continue
                ret = _finite_float((exit_close - entry_close) / entry_close)
                if ret is not None:
                    return_pool.append(ret)

        if len(return_pool) < 10:
            return None

        pool = np.array(return_pool, dtype=float)
        pool = pool[np.isfinite(pool)]
        if len(pool) < 10:
            return None

        # Simulate
        sim_returns: list[float] = []
        for _ in range(self._n_sims):
            picks = rng.choice(pool, size=min(n_signals, len(pool)), replace=True)
            sim_mean = _finite_float(np.nanmean(picks))
            if sim_mean is not None:
                sim_returns.append(sim_mean)

        if not sim_returns:
            return None

        strategy_returns = [
            ret for trade in executed if (ret := _finite_float(trade.return_pct)) is not None
        ]
        if not strategy_returns:
            return None
        strategy_avg = mean(strategy_returns)
        beat_count = sum(1 for r in sim_returns if strategy_avg > r)

        sorted_sims = sorted(sim_returns)
        ci5 = sorted_sims[max(0, int(len(sorted_sims) * 0.05))]
        ci95 = sorted_sims[min(len(sorted_sims) - 1, int(len(sorted_sims) * 0.95))]

        return RandomBaseline(
            mean_return=mean(sim_returns),
            std_return=stdev(sim_returns) if len(sim_returns) > 1 else 0.0,
            ci_lower=ci5,
            ci_upper=ci95,
            beat_fraction=beat_count / self._n_sims,
            n_simulations=self._n_sims,
        )

    def _stress_window_benchmark(
        self,
        trades: list[TradeSignal],
        benchmark_df: pd.DataFrame | None,
    ) -> StressWindowMetrics | None:
        """Evaluate resilience in the benchmark's worst rolling windows."""
        if (
            benchmark_df is None
            or benchmark_df.empty
            or len(benchmark_df) < self._stress_window_days
        ):
            return None
        if "date" not in benchmark_df.columns or "close" not in benchmark_df.columns:
            return None

        benchmark = benchmark_df.copy()
        benchmark["date"] = pd.to_datetime(benchmark["date"]).dt.date
        benchmark = benchmark.sort_values("date").reset_index(drop=True)

        closes = benchmark["close"].to_numpy(dtype=float)
        candidate_windows: list[tuple[int, int, float, float]] = []
        for start in range(0, len(benchmark) - self._stress_window_days + 1):
            end = start + self._stress_window_days - 1
            first_close = closes[start]
            last_close = closes[end]
            if first_close <= 0:
                continue
            window_prices = closes[start : end + 1]
            running_peak = np.maximum.accumulate(window_prices)
            drawdowns = (window_prices - running_peak) / running_peak
            candidate_windows.append(
                (
                    start,
                    end,
                    float((last_close - first_close) / first_close),
                    abs(float(drawdowns.min())),
                )
            )

        if not candidate_windows:
            return None

        selected: list[tuple[int, int, float, float]] = []
        for candidate in sorted(candidate_windows, key=lambda item: (item[2], -item[3])):
            start, end, _, _ = candidate
            overlaps = any(
                not (end < chosen_start or start > chosen_end)
                for chosen_start, chosen_end, _, _ in selected
            )
            if overlaps:
                continue
            selected.append(candidate)
            if len(selected) >= self._stress_window_top_k:
                break

        if not selected:
            return None

        executed = [trade for trade in trades if trade.exit_price is not None]
        cases: list[StressWindowCase] = []
        for idx, (start, end, benchmark_return, benchmark_drawdown) in enumerate(selected, start=1):
            start_date = benchmark["date"].iloc[start]
            end_date = benchmark["date"].iloc[end]
            overlapping = [
                trade
                for trade in executed
                if trade.signal_date <= end_date
                and (trade.exit_date or trade.signal_date) >= start_date
            ]
            returns = [trade.return_pct for trade in overlapping]
            strategy_total_return = (
                float(np.prod(1.0 + np.array(returns, dtype=float)) - 1.0) if returns else 0.0
            )
            strategy_avg_return = float(mean(returns)) if returns else 0.0
            strategy_win_rate = (
                sum(1 for trade in overlapping if trade.return_pct > 0) / len(overlapping)
                if overlapping
                else 0.0
            )
            alpha = strategy_total_return - benchmark_return
            cases.append(
                StressWindowCase(
                    window_num=idx,
                    start_date=start_date,
                    end_date=end_date,
                    benchmark_return=round(benchmark_return, 4),
                    benchmark_drawdown=round(benchmark_drawdown, 4),
                    signal_count=len(overlapping),
                    strategy_win_rate=round(strategy_win_rate, 4),
                    strategy_avg_return=round(strategy_avg_return, 4),
                    strategy_total_return=round(strategy_total_return, 4),
                    alpha=round(alpha, 4),
                )
            )

        worst_case = min(cases, key=lambda case: case.alpha)
        pass_rate = sum(1 for case in cases if case.alpha >= 0.0) / len(cases)
        return StressWindowMetrics(
            window_days=self._stress_window_days,
            top_k=self._stress_window_top_k,
            alpha_pass_threshold=0.0,
            total_windows=len(cases),
            pass_rate=round(pass_rate, 4),
            average_alpha=round(float(mean(case.alpha for case in cases)), 4),
            worst_alpha=round(worst_case.alpha, 4),
            worst_window_start=worst_case.start_date,
            worst_window_end=worst_case.end_date,
            worst_benchmark_return=round(worst_case.benchmark_return, 4),
            windows=cases,
        )
