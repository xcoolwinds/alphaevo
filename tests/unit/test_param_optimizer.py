"""Tests for tunable-parameter optimizer."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from alphaevo.models.enums import MarketType, StrategyCategory
from alphaevo.models.execution import SampleBatch
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
)
from alphaevo.optimizer import (
    ParamOptimizer,
    export_best_param_strategy,
    render_param_optimization_report,
)


def _make_ohlcv(n: int = 80) -> pd.DataFrame:
    rows = []
    price = 100.0
    for idx in range(n):
        price *= 1.004 if idx % 13 < 8 else 0.991
        rows.append(
            {
                "date": date(2025, 1, 1) + timedelta(days=idx),
                "open": round(price * 0.998, 2),
                "high": round(price * 1.012, 2),
                "low": round(price * 0.988, 2),
                "close": round(price, 2),
                "volume": 1_000_000 + idx * 1000,
                "prev_close": round(price / 1.004, 2),
            }
        )
    return pd.DataFrame(rows)


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id="param_optimizer_test_v1",
            name="Param Optimizer Test",
            market=MarketType.US,
            category=StrategyCategory.TREND,
        ),
        description="Always-on strategy for parameter optimizer tests.",
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
        params=StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=rsi_14].value",
                    range=(0, 20),
                    step=10,
                    label="RSI threshold",
                )
            ]
        ),
        market_rules={},
    )


def _make_batch(strategy: Strategy) -> SampleBatch:
    return SampleBatch(
        batch_id="batch",
        strategy_id=strategy.meta.id,
        symbols=["TEST"],
        date_range=(date(2025, 1, 1), date(2025, 3, 31)),
    )


def test_param_optimizer_searches_entry_tunables() -> None:
    strategy = _make_strategy()

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["entry"],
        max_candidates=10,
    )

    assert result.best_candidate is not None
    assert result.evaluation_mode == "fast"
    assert result.tunables_considered == 1
    assert len(result.candidates) > 1
    assert any(
        candidate.target == "entry.conditions[indicator=rsi_14].value"
        for candidate in result.candidates
    )
    assert {candidate.to_value for candidate in result.candidates if candidate.target} == {
        10.0,
        20.0,
    }


def test_param_optimizer_tunes_indicator_periods() -> None:
    strategy = _make_strategy()
    strategy.params.tunable = [
        TunableParam(
            target="entry.conditions[indicator=rsi_14].indicator",
            range=(7, 21),
            step=7,
            label="RSI window",
        )
    ]

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["indicator"],
        max_candidates=10,
    )

    changed_indicators = {
        candidate.strategy.entry.conditions[0].indicator
        for candidate in result.candidates
        if candidate.target is not None
    }
    assert {"rsi_7", "rsi_21"} <= changed_indicators


def test_param_optimizer_tunes_max_holding_days() -> None:
    strategy = _make_strategy()
    strategy.params.tunable = [
        TunableParam(
            target="exit.max_holding_days",
            range=(5, 20),
            step=5,
            label="Max holding days",
        )
    ]

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["exit"],
        max_candidates=10,
    )

    changed_holding_days = {
        candidate.strategy.exit.max_holding_days
        for candidate in result.candidates
        if candidate.target == "exit.max_holding_days"
    }
    assert {5, 15, 20} & changed_holding_days


def test_param_optimizer_includes_range_extremes_early() -> None:
    strategy = _make_strategy()
    strategy.exit.take_profit.value = 3.0
    strategy.params.tunable = [
        TunableParam(
            target="exit.take_profit.value",
            range=(0.8, 4.0),
            step=0.2,
            label="RR",
        )
    ]

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["exit"],
        max_candidates=6,
    )

    changed_values = {candidate.to_value for candidate in result.candidates if candidate.target}
    assert 0.8 in changed_values


def test_param_optimizer_report_mentions_best_candidate() -> None:
    strategy = _make_strategy()
    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["entry"],
        max_candidates=5,
    )

    report = render_param_optimization_report(result)

    assert "# Parameter Optimization Report" in report
    assert "Tunables considered: 1" in report
    assert result.best_candidate_id is not None
    assert result.best_candidate_id in report


def test_param_optimizer_applies_win_rate_gate_and_multi_change_candidates() -> None:
    strategy = _make_strategy()
    strategy.params.tunable.append(
        TunableParam(
            target="exit.take_profit.value",
            range=(1.5, 3.0),
            step=0.5,
            label="RR",
        )
    )

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["params"],
        max_candidates=20,
        max_changes=2,
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
    assert any(len(candidate.changes) == 2 for candidate in result.candidates)
    assert result.qualified_count <= len(result.candidates)


def test_param_optimizer_can_full_evaluate_top_fast_candidates() -> None:
    strategy = _make_strategy()

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["entry"],
        max_candidates=5,
        evaluation_mode="fast",
        full_eval_top_n=2,
    )

    assert result.full_eval_top_n == 2
    assert sum(candidate.evaluation_mode == "full" for candidate in result.candidates) == 2


def test_param_optimizer_does_not_export_failed_gate_candidate(tmp_path: Path) -> None:
    strategy = _make_strategy()

    result = ParamOptimizer(slippage=0.0, commission=0.0, min_data_days=15).optimize(
        strategy,
        {"TEST": _make_ohlcv()},
        _make_batch(strategy),
        spaces=["entry"],
        max_candidates=5,
        min_win_rate=1.0,
        min_avg_return=0.0,
        min_profit_loss_ratio=10.0,
        max_drawdown=0.01,
        min_signals=1,
    )

    assert result.best_candidate is not None
    assert not result.best_candidate.passed_gate
    assert export_best_param_strategy(result, tmp_path) is None
