"""Shared utility functions for AlphaEvo.

Provides lightweight helpers reused across multiple modules.
Domain-heavy logic lives in its respective layer (backtest, evaluator, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone


def utcnow() -> datetime:
    """Return the current UTC datetime (timezone-aware)."""
    return datetime.now(timezone.utc)


def fmt_pct(value: float, decimals: int = 1) -> str:
    """Format a float as a percentage string, e.g. ``0.423 → '42.3%'``."""
    return f"{value * 100:.{decimals}f}%"


def fmt_number(value: float, decimals: int = 2) -> str:
    """Format a number with thousands separator, e.g. ``1234567.8 → '1,234,567.80'``."""
    return f"{value:,.{decimals}f}"


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to ``[lo, hi]``."""
    return max(lo, min(hi, value))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division that returns *default* when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


__all__ = [
    "clamp",
    "fmt_number",
    "fmt_pct",
    "safe_div",
    "utcnow",
]
