"""Tests for exit/risk optimizer."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from alphaevo.models.enums import ExitReason, MarketType, SignalDirection, StrategyCategory
from alphaevo.models.execution import SampleBatch, TradeSignal
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)
from alphaevo.optimizer import (
    ExitOptimizer,
    analyze_exit_points,
    export_best_strategy,
    render_exit_optimization_report,
)


def _make_ohlcv(n: int = 80) -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(n):
        price *= 1.003 if idx % 11 < 7 else 0.992
        rows.append(
            {
                "date": date(2025, 1, 1) + timedelta(days=idx),
                "open": round(price * 0.998, 2),
                "high": round(price * 1.012, 2),
                "low": round(price * 0.988, 2),
                "close": round(price, 2),
                "volume": 1_000_000 + idx * 1000,
                "prev_close": round(price / 1.003, 2),
            }
        )
    return pd.DataFrame(rows)


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="optimizer_test_v1",
            name="Optimizer Test",
            market=MarketType.US,
            category=StrategyCategory.TREND,
        ),
        description="Always-on strategy for optimizer tests.",
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op=">", value=0),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.08),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
        market_rules={},
    )


def test_exit_optimizer_generates_ranked_candidates() -> None:
    strategy = _make_strategy()
    batch = SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )

    result = ExitOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        batch,
        spaces=["exit", "stoploss", "holding"],
        max_candidates=20,
    )

    assert result.best_candidate is not None
    assert result.evaluation_mode == "fast"
    assert len(result.candidates) > 1
    assert result.candidates[0].evaluation.confidence_score >= result.candidates[-1].evaluation.confidence_score
    assert any("exit_triggers=" in change for c in result.candidates for change in c.changes)
    assert result.candidates[0].diagnostics.total_trades >= 0


def test_exit_optimizer_report_mentions_best_candidate() -> None:
    strategy = _make_strategy()
    batch = SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )
    result = ExitOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        batch,
        spaces=["holding"],
        max_candidates=5,
    )

    report = render_exit_optimization_report(result)

    assert "# Exit Optimization Report" in report
    assert "Best Candidate Exit Diagnostics" in report
    assert result.best_candidate_id is not None
    assert result.best_candidate_id in report


def test_exit_optimizer_applies_win_rate_objective_and_gate() -> None:
    strategy = _make_strategy()
    batch = SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )
    result = ExitOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        batch,
        spaces=["holding"],
        max_candidates=5,
        objective="win_rate",
        min_win_rate=0.5,
        min_avg_return=0.0,
        min_profit_loss_ratio=1.0,
        max_drawdown=0.5,
        min_signals=1,
    )

    assert result.objective == "win_rate"
    assert result.evaluation_mode == "fast"
    assert result.min_win_rate == 0.5
    assert result.min_avg_return == 0.0
    assert result.min_profit_loss_ratio == 1.0
    assert result.max_drawdown == 0.5
    assert result.qualified_count <= len(result.candidates)
    assert all(candidate.gate_reasons or candidate.passed_gate for candidate in result.candidates)


def test_exit_optimizer_can_full_evaluate_top_fast_candidates() -> None:
    strategy = _make_strategy()
    batch = SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )
    result = ExitOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        batch,
        spaces=["holding"],
        max_candidates=5,
        evaluation_mode="fast",
        full_eval_top_n=2,
    )

    assert result.full_eval_top_n == 2
    assert sum(candidate.evaluation_mode == "full" for candidate in result.candidates) == 2


def test_exit_optimizer_does_not_export_failed_gate_candidate(tmp_path: Path) -> None:
    strategy = _make_strategy()
    batch = SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )
    result = ExitOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        batch,
        spaces=["holding"],
        max_candidates=5,
        min_win_rate=1.0,
        min_avg_return=0.0,
        min_profit_loss_ratio=10.0,
        max_drawdown=0.01,
        min_signals=1,
    )

    assert result.best_candidate is not None
    assert not result.best_candidate.passed_gate
    assert export_best_strategy(result, tmp_path) is None


def test_analyze_exit_points_flags_late_and_truncated_exits() -> None:
    df = _make_ohlcv()
    signal = TradeSignal(
        symbol="TEST",
        signal_date=df.loc[20, "date"],
        direction=SignalDirection.LONG,
        entry_price=100.0,
        exit_price=105.0,
        exit_date=df.loc[25, "date"],
        exit_reason=ExitReason.TAKE_PROFIT,
        return_pct=0.05,
        holding_days=5,
    )
    df.loc[20:25, "high"] = [101, 103, 110, 109, 108, 106]
    df.loc[26:30, "high"] = [109, 110, 111, 112, 113]

    diagnostics = analyze_exit_points([signal], {"TEST": df})

    assert diagnostics.total_trades == 1
    assert diagnostics.exit_reason_counts["take_profit"] == 1
    assert diagnostics.avg_mfe > 0.05
    assert diagnostics.sold_late_count == 1
    assert diagnostics.take_profit_truncated_count == 1
