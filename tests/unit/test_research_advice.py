"""Tests for deterministic research advice."""

from __future__ import annotations

from types import SimpleNamespace

from alphaevo.evaluator.advice import build_research_advice, render_research_advice
from alphaevo.models.execution import AntiFitMetrics, EvaluationReport, OverallMetrics
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)
from alphaevo.optimizer import ExitDiagnosticSummary


def _strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(id="advice_v1", name="Advice Test", category="trend", market="us"),
        description="test",
        entry=StrategyEntry(
            triggers=[StrategyCondition(indicator="rsi_14", op="<", value=35)]
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
    )


def _evaluation(
    *,
    signals: int = 50,
    win_rate: float = 0.55,
    avg_return: float = 0.01,
    pl: float = 1.5,
    score: float = 0.45,
    anti: AntiFitMetrics | None = None,
) -> EvaluationReport:
    return EvaluationReport(
        strategy_id="advice_v1",
        overall=OverallMetrics(
            signal_count=signals,
            win_rate=win_rate,
            avg_return=avg_return,
            profit_loss_ratio=pl,
            max_drawdown=0.1,
            sharpe_ratio=1.0,
        ),
        anti_overfit=anti or AntiFitMetrics(yearly_consistency=0.5),
        confidence_score=score,
    )


def test_advice_flags_insufficient_evidence() -> None:
    advice = build_research_advice(_strategy(), _evaluation(signals=8), min_signal_count=30)

    assert advice.status == "insufficient_evidence"
    assert advice.recommendations[0].command is not None
    assert "--samples 120" in advice.recommendations[0].command


def test_advice_promotes_stronger_optimized_exit() -> None:
    best = SimpleNamespace(
        candidate_id="advice_v1_opt_001",
        evaluation=_evaluation(score=0.55),
        diagnostics=ExitDiagnosticSummary(total_trades=40, sold_early_count=12),
    )
    optimization = SimpleNamespace(best_candidate=best)

    advice = build_research_advice(
        _strategy(),
        _evaluation(score=0.45),
        optimization=optimization,
    )

    assert advice.status == "promote_optimized_exit"
    assert advice.best_candidate_id == "advice_v1_opt_001"
    assert any("take-profit" in rec.action for rec in advice.recommendations)


def test_advice_promotes_stronger_optimized_param_candidate() -> None:
    best = SimpleNamespace(
        candidate_id="advice_v1_param_001",
        evaluation=_evaluation(score=0.55),
    )
    optimization = SimpleNamespace(best_candidate=best)

    advice = build_research_advice(
        _strategy(),
        _evaluation(score=0.45),
        optimization=optimization,
    )

    assert advice.status == "promote_optimized_exit"
    assert advice.best_candidate_id == "advice_v1_param_001"
    assert "Parameter optimization" in advice.summary


def test_advice_recommends_exit_work_for_weak_payoff() -> None:
    advice = build_research_advice(
        _strategy(),
        _evaluation(win_rate=0.56, avg_return=-0.002, pl=0.8),
    )

    assert advice.status == "optimize_exits"
    assert advice.recommendations[0].command == (
        "alphaevo optimize advice_v1 --spaces exit,stoploss,takeprofit,holding"
    )


def test_render_research_advice_includes_commands() -> None:
    advice = build_research_advice(_strategy(), _evaluation(signals=8), min_signal_count=30)
    md = render_research_advice(advice)

    assert "# Research Advice: advice_v1" in md
    assert "Research output only" in md
    assert "alphaevo run advice_v1" in md
