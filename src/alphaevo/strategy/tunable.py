"""Helpers for resolving and mutating tunable DSL targets."""

from __future__ import annotations

import re
from typing import Any

from alphaevo.models.strategy import Strategy, StrategyCondition

_CONDITION_TARGET_RE = re.compile(
    r"entry\.(conditions|filters|triggers|guards)\[(\d+|indicator=[^]]+)\]\."
    r"(value|indicator(?:\.(?:fast|slow|signal|std))?)"
)
_PERIOD_TARGET_RE = re.compile(r"\.indicator(?:\.(?:fast|slow|signal|std))?$")
_SINGLE_PERIOD_PATTERNS = (
    (r"ma(\d+)", "ma{period}"),
    (r"close_above_ma(\d+)", "close_above_ma{period}"),
    (r"close_below_ma(\d+)", "close_below_ma{period}"),
    (r"close_to_ma(\d+)_pct", "close_to_ma{period}_pct"),
    (r"deviation_from_ma(\d+)_pct", "deviation_from_ma{period}_pct"),
    (r"ma(\d+)_slope", "ma{period}_slope"),
    (r"atr_(\d+)", "atr_{period}"),
    (r"rsi_(\d+)", "rsi_{period}"),
    (r"rsi_(\d+)_zscore", "rsi_{period}_zscore"),
    (r"bollinger_band_width_(\d+)d", "bollinger_band_width_{period}d"),
    (r"price_above_bollinger_upper_(\d+)d", "price_above_bollinger_upper_{period}d"),
    (r"price_below_bollinger_lower_(\d+)d", "price_below_bollinger_lower_{period}d"),
    (r"volume_ratio_1d_(\d+)d", "volume_ratio_1d_{period}d"),
    (r"momentum_(\d+)d", "momentum_{period}d"),
    (r"avg_volume_(\d+)d", "avg_volume_{period}d"),
    (r"days_since_high_(\d+)d", "days_since_high_{period}d"),
    (r"days_since_low_(\d+)d", "days_since_low_{period}d"),
    (r"volatility_(\d+)d", "volatility_{period}d"),
    (r"relative_strength_(\d+)d", "relative_strength_{period}d"),
)
_MACD_PATTERNS = (
    (
        r"macd_histogram_fast(\d+)_slow(\d+)_signal(\d+)",
        "macd_histogram_fast{fast}_slow{slow}_signal{signal}",
    ),
    (
        r"macd_cross_bullish_fast(\d+)_slow(\d+)_signal(\d+)",
        "macd_cross_bullish_fast{fast}_slow{slow}_signal{signal}",
    ),
)
_BOLLINGER_BASE_NAMES = {
    "bollinger_band_width",
    "price_above_bollinger_upper",
    "price_below_bollinger_lower",
}


def resolve_tunable_target(strategy: Strategy, target: str) -> Any:
    """Resolve the current value of a tunable target path."""
    resolved = _resolve_condition_target(strategy, target)
    if resolved is not None:
        condition, field = resolved
        if field == "value":
            return condition.value
        return extract_ma_period(condition.indicator, _period_component(field))

    if target == "exit.stop_loss.value":
        return strategy.exit.stop_loss.value
    if target == "exit.stop_loss.multiplier":
        return strategy.exit.stop_loss.multiplier
    if target == "exit.stop_loss.atr_period":
        if strategy.exit.stop_loss.atr_period is not None:
            return strategy.exit.stop_loss.atr_period
        if strategy.exit.stop_loss.type == "atr":
            return 14
        return None
    if target == "exit.take_profit.value":
        return strategy.exit.take_profit.value
    if target == "exit.take_profit.target":
        tp_target = strategy.exit.take_profit.target
        return extract_ma_period(tp_target) if tp_target else None
    if target == "exit.take_profit.trigger_pct":
        return strategy.exit.take_profit.trigger_pct
    if target == "exit.take_profit.trail_pct":
        return strategy.exit.take_profit.trail_pct
    if target == "exit.max_holding_days":
        return strategy.exit.max_holding_days
    return None


def set_tunable_target(strategy: Strategy, target: str, new_value: Any) -> bool:
    """Set a tunable target path on a strategy in place."""
    resolved = _resolve_condition_target(strategy, target)
    if resolved is not None:
        condition, field = resolved
        if field == "value":
            condition.value = new_value
            return True

        old_indicator = condition.indicator
        component = _period_component(field)
        component_value = _coerce_component_value(new_value, component)
        if component_value is None:
            return False
        new_indicator = replace_ma_period(
            old_indicator,
            component_value,
            component=component,
        )
        if new_indicator is None:
            return False
        condition.indicator = new_indicator
        _retarget_condition_tunables(strategy, old_indicator, new_indicator)
        return True

    if target == "exit.stop_loss.value":
        strategy.exit.stop_loss.value = new_value
        return True
    if target == "exit.stop_loss.multiplier":
        try:
            strategy.exit.stop_loss.multiplier = float(new_value)
        except (TypeError, ValueError):
            return False
        return True
    if target == "exit.stop_loss.atr_period":
        period = _coerce_positive_period(new_value)
        if period is None:
            return False
        strategy.exit.stop_loss.atr_period = period
        return True
    if target == "exit.take_profit.value":
        strategy.exit.take_profit.value = new_value
        return True
    if target == "exit.take_profit.target":
        current = strategy.exit.take_profit.target
        period = _coerce_positive_period(new_value)
        if current is None or period is None:
            return False
        new_target = replace_ma_period(current, period)
        if new_target is None:
            return False
        strategy.exit.take_profit.target = new_target
        return True
    if target == "exit.take_profit.trigger_pct":
        try:
            strategy.exit.take_profit.trigger_pct = float(new_value)
        except (TypeError, ValueError):
            return False
        return True
    if target == "exit.take_profit.trail_pct":
        try:
            strategy.exit.take_profit.trail_pct = float(new_value)
        except (TypeError, ValueError):
            return False
        return True
    if target == "exit.max_holding_days":
        days = _coerce_positive_period(new_value)
        if days is None:
            return False
        strategy.exit.max_holding_days = days
        return True
    return False


def extract_ma_period(value: str | None, component: str | None = None) -> float | int | None:
    """Extract a tunable indicator component value from a supported token."""
    if not value:
        return None

    if component in {None, "single"}:
        if value == "atr":
            return 14
        if value in {
            "bollinger_band_width",
            "price_above_bollinger_upper",
            "price_below_bollinger_lower",
        }:
            return 20
        for pattern, _template in _SINGLE_PERIOD_PATTERNS:
            match = re.fullmatch(pattern, value)
            if match is not None:
                return int(match.group(1))

    bollinger_params = _extract_bollinger_params(value)
    if bollinger_params is not None:
        if component in {None, "single"}:
            return bollinger_params[1]
        if component == "std":
            return bollinger_params[2]
        return None

    macd_periods = _extract_macd_periods(value)
    if macd_periods is not None:
        if component == "fast":
            return macd_periods[0]
        if component == "slow":
            return macd_periods[1]
        if component == "signal":
            return macd_periods[2]
        return None

    dual_periods = _extract_dual_ma_periods(value)
    if dual_periods is None:
        return None
    if component == "fast":
        return dual_periods[0]
    if component == "slow":
        return dual_periods[1]
    return None


def replace_ma_period(
    value: str,
    new_period: float | int,
    *,
    component: str | None = None,
) -> str | None:
    """Replace a tunable indicator period inside a supported token."""
    if component == "std":
        if value_params := _extract_bollinger_params(value):
            base, period, _std = value_params
            std_multiplier = _coerce_positive_float(new_period)
            if std_multiplier is None:
                return None
            return _format_bollinger_indicator(base, period, std_multiplier)
        return None

    int_period = _coerce_positive_period(new_period)
    if int_period is None:
        return None

    if component in {None, "single"}:
        if value_params := _extract_bollinger_params(value):
            base, _period, std_multiplier = value_params
            return _format_bollinger_indicator(base, int_period, std_multiplier)
        if value == "atr":
            return f"atr_{int_period}"
        if value == "bollinger_band_width":
            return f"bollinger_band_width_{int_period}d"
        if value == "price_above_bollinger_upper":
            return f"price_above_bollinger_upper_{int_period}d"
        if value == "price_below_bollinger_lower":
            return f"price_below_bollinger_lower_{int_period}d"
        for pattern, template in _SINGLE_PERIOD_PATTERNS:
            if re.fullmatch(pattern, value) is not None:
                return template.format(period=int_period)

    macd_periods = _extract_macd_periods(value)
    if macd_periods is not None and component in {"fast", "slow", "signal"}:
        fast, slow, signal = macd_periods
        if component == "fast":
            fast = int_period
        elif component == "slow":
            slow = int_period
        else:
            signal = int_period
        if fast < 1 or slow < 1 or signal < 1 or fast >= slow:
            return None
        for pattern, template in _MACD_PATTERNS:
            if re.fullmatch(pattern, value) is not None:
                return template.format(fast=fast, slow=slow, signal=signal)
        if value == "macd_histogram":
            return f"macd_histogram_fast{fast}_slow{slow}_signal{signal}"
        if value == "macd_cross_bullish":
            return f"macd_cross_bullish_fast{fast}_slow{slow}_signal{signal}"
        return None

    dual_periods = _extract_dual_ma_periods(value)
    if dual_periods is None or component not in {"fast", "slow"}:
        return None
    fast, slow = dual_periods
    if component == "fast":
        fast = int_period
    else:
        slow = int_period
    if fast >= slow:
        return None

    if re.fullmatch(r"ma\d+_above_ma\d+", value):
        return f"ma{fast}_above_ma{slow}"
    if re.fullmatch(r"ma\d+_ge_ma\d+_or_crossing", value):
        return f"ma{fast}_ge_ma{slow}_or_crossing"
    return None


def is_period_tunable_target(target: str) -> bool:
    """Return True when a tunable target adjusts an indicator lookback window."""
    return (
        target in {"exit.take_profit.target", "exit.stop_loss.atr_period"}
        or _PERIOD_TARGET_RE.search(target) is not None
    )


def tune_period_value(
    current: float | int,
    step: float,
    target: str,
    *,
    tighten: bool,
    lo: float,
    hi: float,
) -> float | int:
    """Suggest the next indicator component value for a tunable target."""
    if target.endswith(".indicator.fast"):
        raw = current + step if tighten else current - step
    elif target.endswith(".indicator.slow") or target.endswith(".indicator.signal"):
        raw = current - step if tighten else current + step
    elif target.endswith(".indicator.std"):
        raw = current + step if tighten else current - step
    else:
        raw = current - step if tighten else current + step
    clamped = max(lo, min(hi, raw))
    if target.endswith(".indicator.std"):
        return round(float(clamped), 4)
    return int(round(clamped))


def _resolve_condition_target(
    strategy: Strategy,
    target: str,
) -> tuple[StrategyCondition, str] | None:
    """Resolve a condition/filter target path to a concrete condition + field."""
    match = _CONDITION_TARGET_RE.fullmatch(target)
    if match is None:
        return None

    bucket_name, selector, field = match.groups()
    bucket = getattr(strategy.entry, bucket_name)

    if selector.isdigit():
        index = int(selector)
        if index < 0 or index >= len(bucket):
            return None
        return bucket[index], field

    indicator = selector.split("indicator=", 1)[1]
    for condition in bucket:
        if condition.indicator == indicator:
            return condition, field
    return None


def _coerce_positive_period(value: Any) -> int | None:
    """Coerce a value into a positive integer period."""
    if isinstance(value, str):
        period = extract_ma_period(value)
        if period is not None:
            return int(round(period))
    try:
        period = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return period if period > 0 else None


def _coerce_positive_float(value: Any) -> float | None:
    """Coerce a value into a positive float."""
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _coerce_component_value(value: Any, component: str | None) -> float | int | None:
    """Coerce a tunable indicator component based on its semantic type."""
    if component == "std":
        return _coerce_positive_float(value)
    return _coerce_positive_period(value)


def _extract_dual_ma_periods(value: str) -> tuple[int, int] | None:
    """Extract fast/slow periods from supported dual-MA indicators."""
    if match := re.fullmatch(r"ma(\d+)_above_ma(\d+)", value):
        return (int(match.group(1)), int(match.group(2)))
    if match := re.fullmatch(r"ma(\d+)_ge_ma(\d+)_or_crossing", value):
        return (int(match.group(1)), int(match.group(2)))
    return None


def _extract_macd_periods(value: str) -> tuple[int, int, int] | None:
    """Extract fast/slow/signal periods from supported MACD indicators."""
    if value == "macd_histogram" or value == "macd_cross_bullish":
        return (12, 26, 9)
    for pattern, _template in _MACD_PATTERNS:
        if match := re.fullmatch(pattern, value):
            fast = int(match.group(1))
            slow = int(match.group(2))
            signal = int(match.group(3))
            if fast < slow:
                return (fast, slow, signal)
            return None
    return None


def _extract_bollinger_params(value: str) -> tuple[str, int, float] | None:
    """Extract base/period/std multiplier from supported Bollinger indicators."""
    if value in _BOLLINGER_BASE_NAMES:
        return (value, 20, 2.0)
    if match := re.fullmatch(
        r"(bollinger_band_width|price_above_bollinger_upper|price_below_bollinger_lower)"
        r"_(\d+)d(?:_std([0-9]+(?:p[0-9]+)?))?",
        value,
    ):
        period = int(match.group(2))
        std_multiplier = _parse_bollinger_std(match.group(3))
        if period > 0 and std_multiplier is not None:
            return (match.group(1), period, std_multiplier)
    return None


def _parse_bollinger_std(token: str | None) -> float | None:
    """Parse a Bollinger std token like ``2`` or ``1p5``."""
    if token is None:
        return 2.0
    try:
        std_multiplier = float(token.replace("p", "."))
    except ValueError:
        return None
    return std_multiplier if std_multiplier > 0 else None


def _format_bollinger_std(std_multiplier: float) -> str:
    """Format a Bollinger std multiplier into indicator-token form."""
    normalized = f"{std_multiplier:.6f}".rstrip("0").rstrip(".")
    return normalized.replace(".", "p")


def _format_bollinger_indicator(base: str, period: int, std_multiplier: float) -> str:
    """Serialize a Bollinger indicator with period and std multiplier."""
    if std_multiplier == 2.0:
        return f"{base}_{period}d"
    return f"{base}_{period}d_std{_format_bollinger_std(std_multiplier)}"


def _period_component(field: str) -> str | None:
    """Map a tunable field token to the MA period component it refers to."""
    if field == "indicator.fast":
        return "fast"
    if field == "indicator.slow":
        return "slow"
    if field == "indicator.signal":
        return "signal"
    if field == "indicator.std":
        return "std"
    if field == "indicator":
        return "single"
    return None


def is_integer_tunable_target(target: str) -> bool:
    """Return True when a tunable target should be quantized to an integer."""
    return (
        target == "exit.max_holding_days"
        or is_period_tunable_target(target)
        and not target.endswith(".indicator.std")
    )


def _retarget_condition_tunables(strategy: Strategy, old: str, new: str) -> None:
    """Keep indicator-name-based tunable paths aligned after indicator renames."""
    old_token = f"indicator={old}]"
    new_token = f"indicator={new}]"
    for param in strategy.params.tunable:
        if old_token in param.target:
            param.target = param.target.replace(old_token, new_token, 1)
