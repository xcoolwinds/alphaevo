"""Adaptive stock sampler for backtesting.

Selects a representative subset of stocks based on the strategy's universe
constraints and the chosen sampling method (representative, regime-based,
or strategy-scoped).
"""

from __future__ import annotations

import math
import operator
import random
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from alphaevo.models.enums import MarketRegime, SamplingMethod, StrategyCategory
from alphaevo.models.execution import SampleBatch

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphaevo.models.market import StockInfo
    from alphaevo.models.strategy import Strategy, UniverseFilter

# Comparison operators used by universe filters
_OPS: dict[str, Callable[[Any, Any], Any]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

_UNKNOWN_SECTOR = "__unknown__"
_BULLISH_REGIMES = {MarketRegime.TRENDING_UP, MarketRegime.EUPHORIA}
_STRESS_REGIMES = {MarketRegime.TRENDING_DOWN, MarketRegime.PANIC, MarketRegime.VOLATILE}
_REGIME_LOG_CAP_TARGETS: dict[MarketRegime, float] = {
    MarketRegime.TRENDING_UP: 10.6,
    MarketRegime.EUPHORIA: 10.9,
    MarketRegime.RANGE_BOUND: 10.3,
    MarketRegime.VOLATILE: 10.0,
    MarketRegime.TRENDING_DOWN: 9.9,
    MarketRegime.PANIC: 9.6,
}
_CATEGORY_CAP_BIAS: dict[StrategyCategory, float] = {
    StrategyCategory.TREND: 0.15,
    StrategyCategory.ROTATION: 0.1,
    StrategyCategory.REVERSAL: -0.1,
    StrategyCategory.EVENT: -0.2,
    StrategyCategory.FRAMEWORK: 0.0,
}


class AdaptiveSampler:
    """Select stocks for backtesting via stratified / filtered sampling."""

    def __init__(
        self,
        max_symbols: int = 30,
        min_symbols: int = 10,
        seed: int | None = None,
    ) -> None:
        self.max_symbols = max_symbols
        self.min_symbols = min_symbols
        self._rng = random.Random(seed)

    # ── public API ────────────────────────────────────────────────────

    async def sample(
        self,
        strategy: Strategy,
        stock_list: list[StockInfo],
        method: SamplingMethod = SamplingMethod.REPRESENTATIVE,
        date_range: tuple[date, date] | None = None,
        market_regime: MarketRegime | None = None,
    ) -> SampleBatch:
        """Select symbols for backtesting and return a SampleBatch."""
        if date_range is None:
            today = date.today()
            date_range = (today - timedelta(days=365), today)

        if not stock_list:
            return self._make_batch(strategy, [], date_range, method, "empty stock list")

        dispatch = {
            SamplingMethod.REPRESENTATIVE: self._representative_sample,
            SamplingMethod.REGIME_BASED: lambda s, stocks: self._regime_based_sample(
                s,
                stocks,
                market_regime,
            ),
            SamplingMethod.STRATEGY_SCOPED: lambda s, stocks: self._strategy_scoped_sample(
                s,
                stocks,
                market_regime,
            ),
        }
        sampler_fn = dispatch[method]
        symbols = sampler_fn(strategy, stock_list)

        reason = self._sampling_reason(method, strategy, market_regime)

        return self._make_batch(strategy, symbols, date_range, method, reason)

    # ── sampling strategies ───────────────────────────────────────────

    def _representative_sample(self, strategy: Strategy, stocks: list[StockInfo]) -> list[str]:
        """Stratified sampling across sectors and market caps."""
        # Exclude ST stocks
        stocks = [s for s in stocks if not s.is_st]
        if not stocks:
            return []

        sectors: dict[str, list[StockInfo]] = {}
        for s in stocks:
            key = s.sector or _UNKNOWN_SECTOR
            sectors.setdefault(key, []).append(s)

        target = self._clamp(len(stocks))

        if len(sectors) <= 1:
            # No meaningful sector diversity — random sample
            return self._pick_symbols(stocks, target)

        # Proportional allocation per sector, at least 1 per sector
        total = len(stocks)
        selected: list[str] = []
        for sector_stocks in sectors.values():
            n = max(1, round(len(sector_stocks) / total * target))
            # Within each sector, sort by market_cap desc for diversity,
            # then pick evenly spaced entries.
            ordered = sorted(
                sector_stocks,
                key=lambda s: s.market_cap if s.market_cap is not None else 0.0,
                reverse=True,
            )
            picked = self._evenly_spaced(ordered, n)
            selected.extend(p.symbol for p in picked)

        # Trim or pad to target
        if len(selected) > target:
            self._rng.shuffle(selected)
            selected = selected[:target]
        elif len(selected) < self.min_symbols and len(stocks) >= self.min_symbols:
            remaining = [s.symbol for s in stocks if s.symbol not in set(selected)]
            self._rng.shuffle(remaining)
            selected.extend(remaining[: self.min_symbols - len(selected)])

        return selected

    def _strategy_scoped_sample(
        self,
        strategy: Strategy,
        stocks: list[StockInfo],
        market_regime: MarketRegime | None = None,
    ) -> list[str]:
        """Apply universe filters then sample from the filtered set."""
        filtered = self._apply_universe_filters(strategy, stocks)
        if not filtered:
            return []
        if market_regime is not None or strategy.meta.preferred_regime:
            return self._regime_select(strategy, filtered, market_regime)
        return self._representative_sample(strategy, filtered)

    def _regime_based_sample(
        self,
        strategy: Strategy,
        stocks: list[StockInfo],
        market_regime: MarketRegime | None = None,
    ) -> list[str]:
        """Bias sampling toward stocks that better match the current market regime."""
        filtered = self._apply_universe_filters(strategy, stocks)
        return self._regime_select(strategy, filtered, market_regime)

    # ── universe filters ──────────────────────────────────────────────

    def _apply_universe_filters(
        self, strategy: Strategy, stocks: list[StockInfo]
    ) -> list[StockInfo]:
        """Apply ``strategy.universe.filters`` to the stock list."""
        result = list(stocks)
        for f in strategy.universe.filters:
            # If the upstream adapter did not populate this field at all,
            # keep the current pool rather than filtering everything out.
            if not any(getattr(s, f.field, None) is not None for s in result):
                continue
            result = [s for s in result if self._match_filter(s, f)]
        return result

    def _regime_select(
        self,
        strategy: Strategy,
        stocks: list[StockInfo],
        market_regime: MarketRegime | None,
    ) -> list[str]:
        """Select a regime-aware subset using cap/sector heuristics."""
        eligible = [stock for stock in stocks if not stock.is_st]
        if not eligible:
            return []

        active_regime = (
            market_regime or self._preferred_regime(strategy) or MarketRegime.RANGE_BOUND
        )
        target = self._clamp(len(eligible))

        sectors: dict[str, list[StockInfo]] = {}
        for stock in eligible:
            sectors.setdefault(stock.sector or _UNKNOWN_SECTOR, []).append(stock)

        sector_weights = {
            sector: self._sector_weight(strategy, sector_stocks, active_regime)
            for sector, sector_stocks in sectors.items()
        }
        weight_total = sum(sector_weights.values()) or float(len(sectors))

        selected: list[str] = []
        for sector, sector_stocks in sectors.items():
            allocation = max(1, round(sector_weights[sector] / weight_total * target))
            ordered = sorted(
                sector_stocks,
                key=lambda stock: self._regime_fit_score(stock, strategy, active_regime),
                reverse=True,
            )
            selected.extend(stock.symbol for stock in ordered[:allocation])

        symbol_map = {stock.symbol: stock for stock in eligible}
        selected = list(dict.fromkeys(selected))

        if len(selected) > target:
            selected = sorted(
                selected,
                key=lambda symbol: self._regime_fit_score(
                    symbol_map[symbol],
                    strategy,
                    active_regime,
                ),
                reverse=True,
            )[:target]
        elif len(selected) < self.min_symbols and len(eligible) >= self.min_symbols:
            remaining = [
                stock.symbol
                for stock in sorted(
                    eligible,
                    key=lambda stock: self._regime_fit_score(stock, strategy, active_regime),
                    reverse=True,
                )
                if stock.symbol not in set(selected)
            ]
            selected.extend(remaining[: self.min_symbols - len(selected)])

        return selected

    @staticmethod
    def _match_filter(stock: StockInfo, filt: UniverseFilter) -> bool:
        """Return True if *stock* satisfies the filter condition.

        Fields that are ``None`` on the stock are treated as non-matching
        so that stocks without the required data are excluded.
        """
        stock_val = getattr(stock, filt.field, None)
        if stock_val is None:
            return False
        cmp = _OPS.get(filt.op)
        if cmp is None:
            return False
        try:
            return bool(cmp(stock_val, filt.value))
        except TypeError:
            return False

    @staticmethod
    def _preferred_regime(strategy: Strategy) -> MarketRegime | None:
        """Parse the first valid preferred regime from strategy metadata."""
        for value in strategy.meta.preferred_regime:
            try:
                return MarketRegime(value)
            except ValueError:
                continue
        return None

    def _sector_weight(
        self,
        strategy: Strategy,
        sector_stocks: list[StockInfo],
        regime: MarketRegime,
    ) -> float:
        """Compute a sector allocation weight for regime-aware sampling."""
        if not sector_stocks:
            return 0.0
        avg_fit = sum(
            self._regime_fit_score(stock, strategy, regime) for stock in sector_stocks
        ) / len(sector_stocks)
        diversity_bonus = 0.15 if sector_stocks[0].sector else 0.0
        return max(0.1, len(sector_stocks) * (avg_fit + 1.0 + diversity_bonus))

    @staticmethod
    def _regime_fit_score(
        stock: StockInfo,
        strategy: Strategy,
        regime: MarketRegime,
    ) -> float:
        """Score how well a stock fits the active market regime for sampling."""
        if stock.is_st:
            return -10.0

        target_log_cap = _REGIME_LOG_CAP_TARGETS[regime] + _CATEGORY_CAP_BIAS.get(
            strategy.meta.category,
            0.0,
        )
        cap_fit = 0.5
        if stock.market_cap is not None and stock.market_cap > 0:
            log_cap = math.log10(stock.market_cap)
            cap_fit = max(0.0, 1.0 - abs(log_cap - target_log_cap) / 1.5)

        score = cap_fit

        if stock.sector:
            score += 0.15

        if strategy.meta.preferred_regime:
            score += 0.2 if regime.value in strategy.meta.preferred_regime else -0.1

        if strategy.meta.category in {StrategyCategory.TREND, StrategyCategory.ROTATION}:
            if regime in _BULLISH_REGIMES:
                score += 0.2
            if regime in _STRESS_REGIMES:
                score -= 0.05

        if strategy.meta.category in {StrategyCategory.REVERSAL, StrategyCategory.EVENT}:
            if regime in {MarketRegime.VOLATILE, MarketRegime.PANIC, MarketRegime.RANGE_BOUND}:
                score += 0.2
            if regime is MarketRegime.EUPHORIA:
                score -= 0.1

        return score

    # ── helpers ────────────────────────────────────────────────────────

    def _clamp(self, available: int) -> int:
        """Clamp the target sample size between min and max."""
        return max(self.min_symbols, min(self.max_symbols, available))

    def _pick_symbols(self, stocks: list[StockInfo], n: int) -> list[str]:
        """Randomly pick up to *n* symbols."""
        pool = list(stocks)
        self._rng.shuffle(pool)
        return [s.symbol for s in pool[:n]]

    @staticmethod
    def _evenly_spaced(items: list[StockInfo], n: int) -> list[StockInfo]:
        """Pick *n* evenly-spaced items from an ordered list."""
        if n >= len(items):
            return list(items)
        if n <= 0:
            return []
        step = len(items) / n
        return [items[int(i * step)] for i in range(n)]

    def _sampling_reason(
        self,
        method: SamplingMethod,
        strategy: Strategy,
        market_regime: MarketRegime | None,
    ) -> str:
        """Generate a human-readable sampling rationale."""
        if method is SamplingMethod.REPRESENTATIVE:
            return "stratified sampling across sectors"

        current = market_regime.value if market_regime is not None else None
        preferred = (
            ", ".join(strategy.meta.preferred_regime) if strategy.meta.preferred_regime else ""
        )

        if method is SamplingMethod.REGIME_BASED:
            if current and preferred:
                return f"regime-aware sampling for {current} market (preferred: {preferred})"
            if current:
                return f"regime-aware sampling for {current} market"
            if preferred:
                return f"regime-aware sampling using preferred regime ({preferred})"
            return "regime-aware sampling"

        if market_regime is not None or strategy.meta.preferred_regime:
            return "filtered by strategy universe constraints + regime-aware selection"
        return "filtered by strategy universe constraints"

    def _make_batch(
        self,
        strategy: Strategy,
        symbols: list[str],
        date_range: tuple[date, date],
        method: SamplingMethod,
        reason: str,
    ) -> SampleBatch:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        batch_id = f"{strategy.meta.id}_{ts}"
        return SampleBatch(
            batch_id=batch_id,
            strategy_id=strategy.meta.id,
            symbols=symbols,
            date_range=date_range,
            sampling_method=method,
            sampling_reason=reason,
        )
