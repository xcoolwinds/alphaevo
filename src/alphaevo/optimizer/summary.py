"""Human-readable summaries for optimization candidates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from alphaevo.models.strategy import Strategy, StrategyCondition


def format_best_candidate_markdown(
    candidate: Any,
    *,
    title: str = "## Best Strategy Candidate",
) -> list[str]:
    """Return a Markdown block that makes the best candidate easy to inspect."""
    ev = candidate.evaluation.overall
    anti = candidate.evaluation.anti_overfit
    status = "PASS" if candidate.passed_gate else "FAIL"
    lines = [
        title,
        "",
        f"- Candidate: `{candidate.candidate_id}`",
        f"- Strategy ID: `{candidate.strategy.meta.id}`",
        f"- Status: {status}",
        f"- Evaluation mode: {candidate.evaluation_mode}",
        f"- Win rate: {ev.win_rate:.1%}",
        f"- Avg return: {ev.avg_return:.2%}",
        f"- Avg win/loss: {ev.avg_win_return:.2%} / {ev.avg_loss_return:.2%}",
        f"- Total return: {ev.total_return:.2%}",
        f"- P/L ratio: {ev.profit_loss_ratio:.2f}",
        f"- Max drawdown: {ev.max_drawdown:.1%}",
        f"- Signals: {ev.signal_count}",
        f"- Train/Val/Test gaps: {anti.train_val_gap:.1%} / {anti.val_test_gap:.1%}",
        (
            "- Walk-forward gap/pass rate: "
            f"{anti.walk_forward_gap:.1%} / {anti.walk_forward_pass_rate:.1%}"
        ),
        f"- Changes: {_format_changes(candidate.changes)}",
    ]
    if candidate.gate_reasons:
        lines.append(f"- Gate reasons: {', '.join(candidate.gate_reasons)}")
    lines.extend(
        [
            "",
            "### Strategy Rules",
            "",
            f"- Entry triggers: {_format_conditions(candidate.strategy.entry.triggers)}",
            f"- Entry guards: {_format_conditions(candidate.strategy.entry.guards)}",
            f"- Entry conditions: {_format_conditions(candidate.strategy.entry.conditions)}",
            f"- Entry filters: {_format_conditions(candidate.strategy.entry.filters)}",
            f"- Exit triggers: {_format_conditions(candidate.strategy.exit.triggers)}",
            f"- Stop loss: {_format_stop_loss(candidate.strategy)}",
            f"- Take profit: {_format_take_profit(candidate.strategy)}",
            f"- Max holding days: {candidate.strategy.exit.max_holding_days}",
        ]
    )
    return lines


def format_best_candidate_console(candidate: Any) -> str:
    """Return a compact console summary for the best candidate."""
    ev = candidate.evaluation.overall
    anti = candidate.evaluation.anti_overfit
    status = "PASS" if candidate.passed_gate else "FAIL"
    parts = [
        f"Candidate: {candidate.candidate_id}",
        f"Strategy ID: {candidate.strategy.meta.id}",
        f"Status: {status}",
        (
            "Metrics: "
            f"win {ev.win_rate:.1%}, avg {ev.avg_return:.2%}, "
            f"total {ev.total_return:.2%}, P/L {ev.profit_loss_ratio:.2f}, "
            f"DD {ev.max_drawdown:.1%}, signals {ev.signal_count}"
        ),
        (
            "Stability: "
            f"train/val gap {anti.train_val_gap:.1%}, "
            f"val/test gap {anti.val_test_gap:.1%}, "
            f"walk-forward gap {anti.walk_forward_gap:.1%}, "
            f"pass {anti.walk_forward_pass_rate:.1%}"
        ),
        f"Changes: {_format_changes(candidate.changes)}",
        f"Entry: {_format_conditions(candidate.strategy.entry.triggers)}",
        f"Guards: {_format_conditions(candidate.strategy.entry.guards)}",
        f"Exit: {_format_conditions(candidate.strategy.exit.triggers)}",
        f"Risk: {_format_stop_loss(candidate.strategy)}; {_format_take_profit(candidate.strategy)}; max hold {candidate.strategy.exit.max_holding_days}d",
    ]
    if candidate.gate_reasons:
        parts.append(f"Gate reasons: {', '.join(candidate.gate_reasons)}")
    return "\n".join(parts)


def select_high_win_return_candidate(candidates: Iterable[Any]) -> Any | None:
    """Pick the candidate that best balances win rate and return depth."""
    items = list(candidates)
    if not items:
        return None
    return max(items, key=_high_win_return_sort_key)


def _high_win_return_sort_key(
    candidate: Any,
) -> tuple[float, float, float, float, float, float, float]:
    ev = candidate.evaluation.overall
    return (
        1.0 if candidate.passed_gate else 0.0,
        _high_win_return_score(candidate),
        ev.win_rate,
        ev.avg_return,
        ev.total_return,
        ev.profit_loss_ratio,
        -ev.max_drawdown,
    )


def _high_win_return_score(candidate: Any) -> float:
    ev = candidate.evaluation.overall
    win_rate_score = _clamp((ev.win_rate - 0.45) / 0.15)
    avg_return_score = _clamp(ev.avg_return / 0.02)
    total_return_score = _clamp(ev.total_return / 0.50)
    profit_loss_score = _clamp((ev.profit_loss_ratio - 1.0) / 1.5)
    drawdown_score = _clamp(1.0 - ev.max_drawdown / 0.35)
    signal_reliability = _clamp(ev.signal_count / 30.0)
    score = (
        0.30 * win_rate_score
        + 0.30 * avg_return_score
        + 0.25 * total_return_score
        + 0.10 * profit_loss_score
        + 0.05 * drawdown_score
    )
    return round(score * signal_reliability, 6)


def _format_changes(changes: list[str]) -> str:
    return "; ".join(changes) if changes else "baseline"


def _format_conditions(conditions: list[StrategyCondition]) -> str:
    if not conditions:
        return "none"
    return "; ".join(f"{item.indicator} {item.op} {item.value}" for item in conditions)


def _format_stop_loss(strategy: Strategy) -> str:
    stop_loss = strategy.exit.stop_loss
    parts = [f"stop_loss={stop_loss.type}"]
    if stop_loss.value is not None:
        parts.append(f"value={stop_loss.value}")
    if stop_loss.multiplier is not None:
        parts.append(f"multiplier={stop_loss.multiplier}")
    if stop_loss.atr_period is not None:
        parts.append(f"atr_period={stop_loss.atr_period}")
    if stop_loss.reference:
        parts.append(f"reference={stop_loss.reference}")
    if stop_loss.conditions:
        parts.append(f"conditions={_format_conditions(stop_loss.conditions)}")
    return ", ".join(parts)


def _format_take_profit(strategy: Strategy) -> str:
    take_profit = strategy.exit.take_profit
    parts = [f"take_profit={take_profit.type}"]
    if take_profit.value is not None:
        parts.append(f"value={take_profit.value}")
    if take_profit.target:
        parts.append(f"target={take_profit.target}")
    if take_profit.trigger_pct is not None:
        parts.append(f"trigger_pct={take_profit.trigger_pct}")
    if take_profit.trail_pct is not None:
        parts.append(f"trail_pct={take_profit.trail_pct}")
    return ", ".join(parts)


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return min(upper, max(lower, value))
