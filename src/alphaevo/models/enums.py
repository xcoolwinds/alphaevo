"""Enumeration types for AlphaEvo."""

from enum import Enum


class MarketType(str, Enum):
    """Supported market types."""

    A_SHARE = "a_share"
    HK = "hk"
    US = "us"


class StrategyCategory(str, Enum):
    """Strategy classification."""

    TREND = "trend"
    REVERSAL = "reversal"
    EVENT = "event"
    ROTATION = "rotation"
    FRAMEWORK = "framework"


class MarketRegime(str, Enum):
    """Market environment classification."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    VOLATILE = "volatile"
    RANGE_BOUND = "range_bound"
    PANIC = "panic"
    EUPHORIA = "euphoria"


class EvolutionMethod(str, Enum):
    """Strategy evolution method."""

    LLM = "llm"
    PARAM_SEARCH = "param_search"
    HYBRID = "hybrid"


class SamplingMethod(str, Enum):
    """Sample selection method."""

    REPRESENTATIVE = "representative"
    REGIME_BASED = "regime_based"
    STRATEGY_SCOPED = "strategy_scoped"


class StrategyStatus(str, Enum):
    """Strategy lifecycle status."""

    ACTIVE = "active"
    PRUNED = "pruned"
    CHAMPION = "champion"
    DRAFT = "draft"


class SignalDirection(str, Enum):
    """Trade signal direction."""

    LONG = "long"
    SHORT = "short"
    SKIP = "skip"


class ExitReason(str, Enum):
    """Reason for exiting a position."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    MAX_HOLD = "max_hold"
    SIGNAL = "signal"
    MANUAL = "manual"


class ChangeType(str, Enum):
    """Type of strategy modification."""

    TIGHTEN_FILTER = "tighten_filter"
    LOOSEN_FILTER = "loosen_filter"
    ADD_CONDITION = "add_condition"
    REMOVE_CONDITION = "remove_condition"
    ADJUST_EXIT = "adjust_exit"
    CHANGE_UNIVERSE = "change_universe"
    CHANGE_LOGIC = "change_logic"
    DISCOVER_FACTOR = "discover_factor"
