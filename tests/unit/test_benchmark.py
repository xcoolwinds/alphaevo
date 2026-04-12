"""Tests for BenchmarkComparator."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from alphaevo.evaluator.benchmark import BenchmarkComparator, BenchmarkResult
from alphaevo.models.enums import SignalDirection
from alphaevo.models.execution import TradeSignal


def _make_trades(n: int = 20) -> list[TradeSignal]:
    rng = np.random.default_rng(42)
    trades = []
    for i in range(n):
        ret = rng.normal(0.02, 0.05)
        entry = 100.0
        exit_p = entry * (1 + ret)
        d = date(2024, 1, 1) + __import__("datetime").timedelta(days=i)
        trades.append(
            TradeSignal(
                symbol=f"SYM{i % 5}",
                signal_date=d,
                direction=SignalDirection.LONG,
                entry_price=entry,
                exit_price=exit_p,
                exit_date=d + __import__("datetime").timedelta(days=5),
                return_pct=ret,
                holding_days=5,
            )
        )
    return trades


def _make_market_data(n_symbols: int = 5, n_days: int = 100) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(42)
    data = {}
    for i in range(n_symbols):
        close = 100 + np.cumsum(rng.standard_normal(n_days) * 0.5)
        data[f"SYM{i}"] = pd.DataFrame(
            {
                "open": close + rng.uniform(-0.3, 0.3, n_days),
                "high": close + abs(rng.standard_normal(n_days)),
                "low": close - abs(rng.standard_normal(n_days)),
                "close": close,
                "volume": rng.integers(1000, 5000, n_days).astype(float),
            }
        )
    return data


def _make_benchmark_df(n_days: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    close = np.concatenate(
        [
            np.linspace(100, 102, 20),
            np.linspace(102, 78, 20),
            np.linspace(78, 82, 20),
            np.linspace(82, 88, 20),
        ]
    )[:n_days]
    return pd.DataFrame({"date": dates.date, "close": close})


class TestBenchmarkComparator:
    def test_compare_produces_result(self):
        trades = _make_trades()
        data = _make_market_data()
        comp = BenchmarkComparator(n_random_simulations=50, seed=42)
        result = comp.compare(trades, data)
        assert isinstance(result, BenchmarkResult)
        assert result.buy_hold.symbols_used == 5
        assert result.summary

    def test_buy_hold_returns(self):
        data = _make_market_data()
        comp = BenchmarkComparator()
        bh = comp._buy_and_hold([], data)
        assert bh.symbols_used == 5
        # Buy-and-hold return should be some value based on generated data
        assert isinstance(bh.benchmark_return, float)

    def test_random_baseline_computed(self):
        trades = _make_trades(30)
        data = _make_market_data()
        comp = BenchmarkComparator(n_random_simulations=50, seed=42)
        result = comp.compare(trades, data)
        assert result.random_baseline is not None
        assert result.random_baseline.n_simulations == 50
        assert 0.0 <= result.random_baseline.beat_fraction <= 1.0

    def test_empty_trades(self):
        data = _make_market_data()
        comp = BenchmarkComparator()
        result = comp.compare([], data)
        assert result.random_baseline is None

    def test_empty_data(self):
        trades = _make_trades()
        comp = BenchmarkComparator()
        result = comp.compare(trades, {})
        assert result.buy_hold.symbols_used == 0

    def test_excess_return_sign(self):
        trades = _make_trades()
        data = _make_market_data()
        comp = BenchmarkComparator()
        result = comp.compare(trades, data)
        expected = result.buy_hold.strategy_return - result.buy_hold.benchmark_return
        assert abs(result.buy_hold.excess_return - expected) < 1e-10

    def test_stress_windows_computed(self):
        trades = _make_trades(30)
        data = _make_market_data()
        benchmark_df = _make_benchmark_df()
        comp = BenchmarkComparator(stress_window_days=10, stress_window_top_k=2)
        result = comp.compare(trades, data, benchmark_df=benchmark_df)
        assert result.stress_windows is not None
        assert result.stress_windows.window_days == 10
        assert result.stress_windows.total_windows == 2
        assert len(result.stress_windows.windows) == 2
