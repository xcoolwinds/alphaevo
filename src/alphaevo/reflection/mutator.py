"""Strategy mutator — applies proposed changes to produce a new strategy version.

Pure logic module (no LLM). Takes a Strategy and a list of StrategyChange objects
and produces a new Strategy with incremented version and updated parent_id.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import StrategyChange
from alphaevo.models.strategy import (
    Strategy,
    StrategyCondition,
)
from alphaevo.strategy.tunable import set_tunable_target

logger = logging.getLogger(__name__)


class MutationError(Exception):
    """Raised when a mutation cannot be applied."""


class StrategyMutator:
    """Apply StrategyChange objects to mutate a strategy into a new version.

    Safety guardrails (from AGENTS.md §11):
    - max_changes: maximum changes per round (default 3)
    - complexity_limit: max entry conditions (default 8)
    """

    def __init__(
        self,
        max_changes: int = 3,
        complexity_limit: int = 8,
    ) -> None:
        if complexity_limit < 1:
            raise ValueError(f"complexity_limit must be >= 1, got {complexity_limit}")
        self.max_changes = max_changes
        self.complexity_limit = complexity_limit

    def mutate(
        self,
        strategy: Strategy,
        changes: list[StrategyChange],
        *,
        atomic: bool = False,
    ) -> Strategy:
        """Apply changes to strategy, returning a new version.

        When ``atomic`` is True, every change in the bundle must apply
        successfully or the whole mutation is rejected.

        Raises MutationError if changes violate safety guardrails.
        """
        if len(changes) > self.max_changes:
            if atomic:
                raise MutationError(
                    f"Atomic mutation requires <= {self.max_changes} changes, got {len(changes)}"
                )
            logger.warning("Truncating %d changes to max %d", len(changes), self.max_changes)
            changes = changes[: self.max_changes]

        # Deep copy to avoid modifying original
        new = strategy.model_copy(deep=True)

        applied: list[StrategyChange] = []
        for change in changes:
            try:
                self._apply_change(new, change)
                applied.append(change)
            except Exception as e:
                if atomic:
                    raise MutationError(
                        f"Atomic mutation failed on {change.target}: {e}"
                    ) from e
                logger.warning("Skipping change %s: %s", change.target, e)

        if not applied:
            raise MutationError("No changes could be applied")

        # Enforce complexity limit on total (conditions + filters)
        n_conditions = len(new.entry.conditions) + len(new.entry.filters)
        if n_conditions > self.complexity_limit:
            if atomic:
                raise MutationError(
                    f"Atomic mutation would exceed complexity limit "
                    f"({n_conditions} > {self.complexity_limit})"
                )
            logger.warning(
                "Complexity %d exceeds limit %d, trimming",
                n_conditions,
                self.complexity_limit,
            )
            # Trim conditions first (entry signals are easier to regenerate),
            # then filters only if conditions alone can't bring us under limit.
            if len(new.entry.conditions) > self.complexity_limit:
                new.entry.conditions = new.entry.conditions[: self.complexity_limit]
                new.entry.filters = []
            else:
                max_filters = self.complexity_limit - len(new.entry.conditions)
                new.entry.filters = new.entry.filters[:max_filters]

        # Bump version
        old_id = new.meta.id
        new_version = new.meta.version + 1
        base_name = old_id.rsplit("_v", 1)[0]
        new_id = f"{base_name}_v{new_version}"

        new.meta.id = new_id
        new.meta.version = new_version
        new.meta.parent_id = old_id
        new.meta.created_at = datetime.now(timezone.utc)

        return new

    def _apply_change(self, strategy: Strategy, change: StrategyChange) -> None:
        """Apply a single change to a strategy (in-place)."""
        handler = _CHANGE_HANDLERS.get(change.change_type)
        if handler is None:
            raise MutationError(f"Unknown change type: {change.change_type}")
        handler(strategy, change)


# ── Change handlers ──────────────────────────────────────────────────


def _tighten_filter(strategy: Strategy, change: StrategyChange) -> None:
    """Make a condition more restrictive (e.g., lower threshold)."""
    _update_condition_value(strategy, change)


def _loosen_filter(strategy: Strategy, change: StrategyChange) -> None:
    """Make a condition less restrictive (e.g., raise threshold)."""
    _update_condition_value(strategy, change)


def _add_condition(strategy: Strategy, change: StrategyChange) -> None:
    """Add a new entry condition."""
    if not isinstance(change.to_value, dict):
        raise MutationError(f"ADD_CONDITION requires to_value as dict, got {type(change.to_value)}")
    new_cond = StrategyCondition(**change.to_value)

    # Validate that the indicator exists in the registry before adding,
    # so LLM-invented indicators are caught before they reach backtest.
    from alphaevo.backtest.indicators import IndicatorRegistry

    if not IndicatorRegistry.is_registered(new_cond.indicator):
        raise MutationError(
            f"Unknown indicator '{new_cond.indicator}' proposed by LLM. "
            f"Available: {IndicatorRegistry.available()}"
        )

    if change.target.startswith("entry.filters"):
        strategy.entry.filters.append(new_cond)
    else:
        strategy.entry.conditions.append(new_cond)


def _remove_condition(strategy: Strategy, change: StrategyChange) -> None:
    """Remove an entry condition by indicator name."""
    indicator = change.target.split("indicator=")[-1].split("]")[0].strip()
    if not indicator:
        raise MutationError(f"Invalid target format (empty indicator name): {change.target}")

    # Try conditions first
    for i, cond in enumerate(strategy.entry.conditions):
        if cond.indicator == indicator:
            strategy.entry.conditions.pop(i)
            return

    # Try filters
    for i, filt in enumerate(strategy.entry.filters):
        if filt.indicator == indicator:
            strategy.entry.filters.pop(i)
            return

    logger.warning("Condition with indicator=%s not found, skipping", indicator)


def _adjust_exit(strategy: Strategy, change: StrategyChange) -> None:
    """Modify exit parameters (stop loss, take profit, max holding days)."""
    target = change.target
    val = _sanitize_condition_value(change.to_value)
    if "stop_loss.value" in target and val is not None:
        strategy.exit.stop_loss.value = float(val)
    elif "stop_loss.type" in target and val is not None:
        new_type = str(val)
        strategy.exit.stop_loss.type = new_type
        if new_type == "atr":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.reference = None
            strategy.exit.stop_loss.conditions = None
        elif new_type == "price_level":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.conditions = None
        elif new_type == "composite":
            strategy.exit.stop_loss.value = None
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.reference = None
        else:
            strategy.exit.stop_loss.multiplier = None
            strategy.exit.stop_loss.atr_period = None
            strategy.exit.stop_loss.reference = None
            strategy.exit.stop_loss.conditions = None
    elif "stop_loss.reference" in target and val is not None:
        strategy.exit.stop_loss.reference = str(val)
    elif "stop_loss.multiplier" in target and val is not None:
        strategy.exit.stop_loss.multiplier = float(val)
    elif "stop_loss.atr_period" in target and val is not None:
        if not set_tunable_target(strategy, target, val):
            raise MutationError(f"Invalid stop_loss.atr_period update: {change.target}")
    elif "take_profit.value" in target and val is not None:
        strategy.exit.take_profit.value = float(val)
    elif "take_profit.target" in target and val is not None:
        if not set_tunable_target(strategy, target, val):
            raise MutationError(f"Invalid take_profit.target update: {change.target}")
    elif "take_profit.type" in target and val is not None:
        new_type = str(val)
        strategy.exit.take_profit.type = new_type
        if new_type == "target_ma":
            strategy.exit.take_profit.value = None
            strategy.exit.take_profit.trigger_pct = None
            strategy.exit.take_profit.trail_pct = None
        elif new_type == "trailing":
            strategy.exit.take_profit.value = None
            strategy.exit.take_profit.target = None
        else:
            strategy.exit.take_profit.target = None
            strategy.exit.take_profit.trigger_pct = None
            strategy.exit.take_profit.trail_pct = None
    elif "take_profit.trigger_pct" in target and val is not None:
        strategy.exit.take_profit.trigger_pct = float(val)
    elif "take_profit.trail_pct" in target and val is not None:
        strategy.exit.take_profit.trail_pct = float(val)
    elif "max_holding_days" in target and val is not None:
        strategy.exit.max_holding_days = int(val)
    else:
        raise MutationError(f"Unknown exit target: {target}")


def _change_universe(strategy: Strategy, change: StrategyChange) -> None:
    """Modify universe filters."""
    if "market" in change.target and isinstance(change.to_value, list):
        strategy.universe.market = change.to_value
    else:
        logger.warning("Universe change target '%s' not supported", change.target)


def _change_logic(strategy: Strategy, change: StrategyChange) -> None:
    """Switch entry logic between 'and' and 'or'."""
    new_logic = str(change.to_value).lower()
    if new_logic not in ("and", "or"):
        raise MutationError(f"Invalid logic value: {change.to_value}, expected 'and' or 'or'")
    strategy.entry.logic = new_logic  # type: ignore[assignment]


def _discover_factor(strategy: Strategy, change: StrategyChange) -> None:
    """Add a newly discovered factor as a condition.

    The factor must already be registered in IndicatorRegistry by the
    AlphaFactory pipeline before this handler is called.
    """
    if not isinstance(change.to_value, dict):
        raise MutationError(
            f"DISCOVER_FACTOR requires to_value as dict with condition fields, "
            f"got {type(change.to_value)}"
        )
    new_cond = StrategyCondition(**change.to_value)

    from alphaevo.backtest.indicators import IndicatorRegistry

    if not IndicatorRegistry.is_registered(new_cond.indicator):
        raise MutationError(
            f"Discovered factor '{new_cond.indicator}' not registered in IndicatorRegistry. "
            f"Ensure AlphaFactory.discover() ran successfully first."
        )

    if change.target.startswith("entry.filters"):
        strategy.entry.filters.append(new_cond)
    else:
        strategy.entry.conditions.append(new_cond)


def _sanitize_condition_value(raw: Any) -> Any:
    """Strip operator prefixes LLMs may include (e.g. '< 65.0' → 65.0).

    LLMs sometimes output to_value as '"< 65.0"' or '">= 1.5"' instead of
    the bare number.  We detect the pattern and extract the numeric part.
    Boolean strings ('true'/'false') are also normalised.
    """
    if not isinstance(raw, str):
        return raw
    # Strip leading comparison operators: <, <=, >, >=, ==, !=
    stripped = re.sub(r"^[<>=!]+\s*", "", raw).strip()
    if not stripped:
        return raw
    # Boolean
    if stripped.lower() in ("true", "false"):
        return stripped.lower() == "true"
    # Numeric
    try:
        return int(stripped) if "." not in stripped else float(stripped)
    except ValueError:
        return raw


def _update_condition_value(strategy: Strategy, change: StrategyChange) -> None:
    """Update the value of an existing condition by indicator name."""
    sanitized_value = _sanitize_condition_value(change.to_value)
    if set_tunable_target(strategy, change.target, sanitized_value):
        return

    indicator = change.target.split("indicator=")[-1].split("]")[0]

    for cond in strategy.entry.conditions + strategy.entry.filters:
        if cond.indicator == indicator:
            if sanitized_value is not None:
                cond.value = sanitized_value
            return

    raise MutationError(f"Condition with indicator={indicator} not found")


_CHANGE_HANDLERS = {
    ChangeType.TIGHTEN_FILTER: _tighten_filter,
    ChangeType.LOOSEN_FILTER: _loosen_filter,
    ChangeType.ADD_CONDITION: _add_condition,
    ChangeType.REMOVE_CONDITION: _remove_condition,
    ChangeType.ADJUST_EXIT: _adjust_exit,
    ChangeType.CHANGE_UNIVERSE: _change_universe,
    ChangeType.CHANGE_LOGIC: _change_logic,
    ChangeType.DISCOVER_FACTOR: _discover_factor,
}
