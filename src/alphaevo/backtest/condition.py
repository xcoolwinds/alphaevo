"""Condition evaluator — evaluates StrategyCondition against indicator values.

Bridges the gap between YAML DSL conditions and the IndicatorRegistry.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from alphaevo.backtest.indicators import IndicatorRegistry

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

    from alphaevo.models.market import IndicatorContext
    from alphaevo.models.strategy import StrategyCondition, StrategyEntry

# ── Comparison operators ─────────────────────────────────────────────

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


class ConditionEvaluator:
    """Evaluate strategy conditions against computed indicator values."""

    def evaluate_condition(
        self,
        condition: StrategyCondition,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> bool:
        """Evaluate a single condition: compute indicator then compare."""
        if not IndicatorRegistry.is_registered(condition.indicator):
            # Unknown indicator → conservative block.
            # The engine pre-validates via _collect_unknown_indicators, so
            # reaching here only happens if validation was bypassed.
            return False

        actual = IndicatorRegistry.compute(condition.indicator, df, idx, ctx)
        expected = condition.value
        op_fn = _OPS.get(condition.op)
        if op_fn is None:
            return False  # Unknown operator → block

        # Type coercion: align actual with expected type
        try:
            if isinstance(expected, bool):
                actual = bool(actual)
            elif isinstance(expected, (int, float)):
                actual = float(actual)
                expected = float(expected)
        except (TypeError, ValueError):
            return False

        return bool(op_fn(actual, expected))

    def evaluate_group(
        self,
        conditions: list[StrategyCondition],
        logic: str,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> bool:
        """Evaluate a group of conditions with AND/OR logic."""
        if not conditions:
            return True

        if logic == "or":
            return any(self.evaluate_condition(c, df, idx, ctx) for c in conditions)
        # Default: AND
        return all(self.evaluate_condition(c, df, idx, ctx) for c in conditions)

    def evaluate_entry(
        self,
        entry: StrategyEntry,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> bool:
        """Evaluate full entry: conditions (AND/OR) + filters (always AND)."""
        cond_ok = self.evaluate_group(entry.conditions, entry.logic, df, idx, ctx)
        if not cond_ok:
            return False
        # Filters are always AND
        return self.evaluate_group(entry.filters, "and", df, idx, ctx)
