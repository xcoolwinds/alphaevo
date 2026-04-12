"""Backtest engine — simulates strategy execution on historical data.

Core loop: for each symbol in sample batch, iterate daily bars,
evaluate entry conditions, manage positions, check exits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import pandas as pd

from alphaevo.backtest.condition import ConditionEvaluator
from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.backtest.rules import MarketRuleChecker
from alphaevo.models.enums import ExitReason, MarketRegime, SignalDirection
from alphaevo.models.execution import BacktestResult, SampleBatch, TradeSignal
from alphaevo.sampler.regime import RegimeDetector

if TYPE_CHECKING:
    from alphaevo.models.market import IndicatorContext
    from alphaevo.models.strategy import MarketRuleConfig, StopLossConfig, Strategy


@dataclass
class _Position:
    """Internal position tracking during backtest."""

    symbol: str
    entry_idx: int
    entry_date: date
    entry_price: float
    regime: MarketRegime | None = None
    sector: str | None = None
    indicator_snapshot: dict[str, float | bool] | None = None


class BacktestEngine:
    """Signal-level backtest engine.

    Iterates daily bars for each symbol, evaluates entry/exit conditions,
    and records trade signals with outcomes.
    """

    def __init__(
        self,
        condition_evaluator: ConditionEvaluator | None = None,
        rule_checker: MarketRuleChecker | None = None,
        slippage: float = 0.001,
        commission: float = 0.0003,
        min_data_days: int = 30,
    ) -> None:
        if not 0 <= slippage <= 0.1:
            raise ValueError(f"slippage must be in [0, 0.1], got {slippage}")
        if not 0 <= commission <= 0.1:
            raise ValueError(f"commission must be in [0, 0.1], got {commission}")
        self.evaluator = condition_evaluator or ConditionEvaluator()
        self.rule_checker = rule_checker or MarketRuleChecker()
        self.slippage = slippage
        self.commission = commission
        self.min_data_days = min_data_days

    def run(
        self,
        strategy: Strategy,
        data: dict[str, pd.DataFrame],
        batch: SampleBatch,
        contexts: dict[str, IndicatorContext] | None = None,
    ) -> BacktestResult:
        """Run backtest for all symbols in batch.

        Args:
            strategy: Strategy to test.
            data: Dict of symbol → OHLCV DataFrame.
                  DataFrame must have columns: date, open, high, low, close, volume.
                  Optionally: prev_close, amount.
            batch: Sample batch definition.
            contexts: Optional per-symbol IndicatorContext for L2/L3 indicators.

        Returns:
            BacktestResult with all trade signals.
        """
        unknown_indicators = self._collect_unknown_indicators(strategy)
        if unknown_indicators:
            names = ", ".join(unknown_indicators)
            raise ValueError(f"Unknown indicators in strategy: {names}")
        self._regime_cache: dict[tuple[int, int], MarketRegime | None] = {}

        # Resolve market rules
        market_key = strategy.meta.market.value
        rules = strategy.market_rules.get(
            market_key,
            MarketRuleChecker.default_rules(market_key),
        )

        # Resolve entry timing and per-strategy slippage override
        timing = "next_open"
        effective_slippage = self.slippage
        if strategy.entry.execution:
            timing = strategy.entry.execution.timing
            if strategy.entry.execution.slippage is not None:
                effective_slippage = strategy.entry.execution.slippage

        all_signals: list[TradeSignal] = []

        for symbol in batch.symbols:
            df = data.get(symbol)
            if df is None or df.empty or len(df) < self.min_data_days:
                continue

            ctx = contexts.get(symbol) if contexts else None
            signals = self._run_symbol(
                strategy,
                df,
                symbol,
                rules,
                ctx,
                timing,
                effective_slippage,
            )
            all_signals.extend(signals)

        executed = [s for s in all_signals if s.exit_price is not None]
        return BacktestResult(
            strategy_id=strategy.meta.id,
            batch_id=batch.batch_id,
            signals=all_signals,
            total_signals=len(all_signals),
            executed_signals=len(executed),
            skipped_signals=len(all_signals) - len(executed),
            date_range=batch.date_range,
        )

    def _run_symbol(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        symbol: str,
        rules: MarketRuleConfig,
        ctx: IndicatorContext | None,
        timing: str,
        slippage: float,
    ) -> list[TradeSignal]:
        """Run backtest on a single symbol."""
        signals: list[TradeSignal] = []
        position: _Position | None = None
        # Ensure we have enough bars for the largest indicator window
        warmup = _indicator_warmup(self._collect_all_indicators(strategy))
        start_idx = max(self.min_data_days, warmup)

        for idx in range(start_idx, len(df)):
            # ── Holding a position: check exit ──
            if position is not None:
                if not self.rule_checker.can_sell(df, idx, position.entry_idx, rules):
                    continue

                exit_result = self._check_exit(position, strategy, df, idx, ctx)
                if exit_result is not None:
                    reason, exit_price = exit_result
                    signal = self._close_position(position, df, idx, exit_price, reason, slippage)
                    signals.append(signal)
                    position = None
                continue  # Don't check entry while holding

            # ── No position: check entry ──
            if not self.rule_checker.can_buy(df, idx, rules):
                continue

            if not self._regime_allows_entry(strategy, df, idx, ctx):
                continue

            if self.evaluator.evaluate_entry(strategy.entry, df, idx, ctx):
                position = self._open_position(symbol, df, idx, timing, slippage, ctx)
                if position is None:
                    continue
                # Capture indicator snapshot for failure analysis
                position.indicator_snapshot = self._capture_indicators(
                    strategy,
                    df,
                    idx,
                    ctx,
                )

        # Force close any remaining position at end
        if position is not None:
            signal = self._close_position(
                position, df, len(df) - 1, df["close"].iloc[-1], ExitReason.MAX_HOLD, slippage
            )
            signals.append(signal)

        return signals

    def _open_position(
        self,
        symbol: str,
        df: pd.DataFrame,
        signal_idx: int,
        timing: str,
        slippage: float,
        ctx: IndicatorContext | None,
    ) -> _Position | None:
        """Create a new position based on entry timing."""
        if timing == "next_open" and signal_idx + 1 < len(df):
            entry_idx = signal_idx + 1
            entry_price = df["open"].iloc[entry_idx]
        elif timing == "close":
            entry_idx = signal_idx
            entry_price = df["close"].iloc[signal_idx]
        elif timing == "breakout_high":
            if signal_idx + 1 >= len(df):
                return None  # no next bar to check breakout
            entry_idx = signal_idx + 1
            breakout_price = df["high"].iloc[signal_idx]
            # Confirm breakout: next day must open above signal day's high
            next_open = df["open"].iloc[entry_idx]
            if next_open < breakout_price:
                return None
            entry_price = next_open
        else:
            # Default: next open
            entry_idx = min(signal_idx + 1, len(df) - 1)
            entry_price = df["open"].iloc[entry_idx]

        # Apply slippage (buy slightly higher)
        entry_price *= 1 + slippage

        entry_date = self._get_date(df, entry_idx)
        return _Position(
            symbol=symbol,
            entry_idx=entry_idx,
            entry_date=entry_date,
            entry_price=entry_price,
            regime=self._get_current_regime(df, signal_idx, ctx),
            sector=self._get_sector(ctx),
        )

    def _check_exit(
        self,
        position: _Position,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> tuple[ExitReason, float] | None:
        """Check all exit conditions. Returns (reason, exit_price) or None."""
        exit_cfg = strategy.exit
        current_price = df["close"].iloc[idx]
        holding_days = idx - position.entry_idx

        # 1. Max holding days
        if holding_days >= exit_cfg.max_holding_days:
            return (ExitReason.MAX_HOLD, current_price)

        # 2. Stop loss
        sl = exit_cfg.stop_loss
        if sl.type == "pct" and sl.value is not None:
            stop_price = position.entry_price * (1 - sl.value)
            low_price = df["low"].iloc[idx]
            # Use intraday low for more realistic stop-loss simulation
            if low_price <= stop_price:
                return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "atr":
            atr_ind = f"atr_{sl.atr_period}" if sl.atr_period and sl.atr_period != 14 else "atr"
            atr_val = IndicatorRegistry.compute(atr_ind, df, idx, ctx)
            multiplier = sl.multiplier or 2.0
            stop_price = position.entry_price - atr_val * multiplier
            low_price = df["low"].iloc[idx]
            if low_price <= stop_price:
                return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "pct_from_low" and sl.value is not None:
            # Trailing stop: exit when low drops sl.value% from highest high
            # Start from bar after entry to avoid using entry bar's high as anchor
            trail_start = min(position.entry_idx + 1, idx)
            high_since_entry = df["high"].iloc[trail_start : idx + 1].max()
            low_price = df["low"].iloc[idx]
            if high_since_entry > 0:
                drawdown = (low_price - high_since_entry) / high_since_entry
                if drawdown <= -sl.value:
                    # Exit at the stop price, not the close
                    stop_price = high_since_entry * (1 - sl.value)
                    return (ExitReason.STOP_LOSS, stop_price)

        elif sl.type == "price_level":
            price_level = self._resolve_price_level(position, sl, df, idx, ctx)
            if price_level is not None and df["low"].iloc[idx] <= price_level:
                return (ExitReason.STOP_LOSS, price_level)

        elif sl.type == "composite" and sl.conditions:
            # Any condition triggers stop loss
            for cond in sl.conditions:
                if self.evaluator.evaluate_condition(cond, df, idx, ctx):
                    return (ExitReason.STOP_LOSS, current_price)

        # 3. Take profit
        tp = exit_cfg.take_profit
        risk_amount = self._compute_risk(position, sl, df, idx, ctx)

        if tp.type == "rr" and tp.value is not None:
            target = position.entry_price + risk_amount * tp.value
            # Use intraday high for more realistic take-profit
            if df["high"].iloc[idx] >= target:
                return (ExitReason.TAKE_PROFIT, target)

        elif tp.type == "pct" and tp.value is not None:
            target = position.entry_price * (1 + tp.value)
            if df["high"].iloc[idx] >= target:
                return (ExitReason.TAKE_PROFIT, target)

        elif tp.type == "target_ma" and tp.target:
            target_ma = self._resolve_target_ma(tp.target, df, idx)
            if target_ma is not None and df["high"].iloc[idx] >= target_ma:
                return (ExitReason.TAKE_PROFIT, target_ma)

        elif tp.type == "trailing":
            trigger_pct = tp.trigger_pct or 0.08
            trail_pct = tp.trail_pct or 0.04
            high_since = df["high"].iloc[position.entry_idx : idx + 1].max()
            gain_from_entry = (high_since - position.entry_price) / position.entry_price
            if gain_from_entry >= trigger_pct:
                low_price = df["low"].iloc[idx]
                drawdown_from_high = (high_since - low_price) / high_since
                if drawdown_from_high >= trail_pct:
                    # Exit at the trail level, not the close
                    trail_price = high_since * (1 - trail_pct)
                    return (ExitReason.TAKE_PROFIT, trail_price)

        return None

    def _compute_risk(
        self,
        position: _Position,
        sl_config: StopLossConfig,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> float:
        """Compute per-share risk amount for RR-based take profit."""
        if sl_config.type in ("pct", "pct_from_low") and sl_config.value is not None:
            return float(abs(position.entry_price * sl_config.value))
        elif sl_config.type == "atr":
            atr_ind = f"atr_{sl_config.atr_period}" if sl_config.atr_period and sl_config.atr_period != 14 else "atr"
            atr_val = IndicatorRegistry.compute(atr_ind, df, idx, ctx)
            return float(atr_val * (sl_config.multiplier or 2.0))
        # Default: 5% risk (composite or unknown types)
        return float(position.entry_price * 0.05)

    def _close_position(
        self,
        position: _Position,
        df: pd.DataFrame,
        idx: int,
        exit_price: float,
        reason: ExitReason,
        slippage: float,
    ) -> TradeSignal:
        """Close a position and create a TradeSignal."""
        # Apply slippage (sell slightly lower)
        exit_price *= 1 - slippage
        # Apply commission both ways
        net_entry = position.entry_price * (1 + self.commission)
        net_exit = exit_price * (1 - self.commission)
        return_pct = (net_exit - net_entry) / net_entry

        exit_date = self._get_date(df, idx)
        holding_days = idx - position.entry_idx

        return TradeSignal(
            symbol=position.symbol,
            signal_date=position.entry_date,
            direction=SignalDirection.LONG,
            entry_price=position.entry_price,
            exit_price=exit_price,
            exit_date=exit_date,
            exit_reason=reason,
            return_pct=round(return_pct, 6),
            holding_days=holding_days,
            regime=position.regime,
            sector=position.sector,
            indicator_snapshot=position.indicator_snapshot or {},
        )

    @staticmethod
    def _resolve_target_ma(target: str, df: pd.DataFrame, idx: int) -> float | None:
        """Resolve a take-profit moving-average target."""
        target = target.lower().strip()
        periods = {
            "ma5": 5,
            "ma10": 10,
            "ma20": 20,
            "ma60": 60,
        }
        period = periods.get(target)
        if period is None or idx + 1 < period:
            return None
        window = df["close"].iloc[idx - period + 1 : idx + 1]
        return float(window.mean())

    @staticmethod
    def _resolve_price_level(
        position: _Position,
        sl_config: StopLossConfig,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None = None,
    ) -> float | None:
        """Resolve a stop-loss price level from config."""
        if sl_config.value is not None:
            return float(sl_config.value)

        reference = (sl_config.reference or "").strip()
        if not reference:
            return None

        if reference in {"entry", "entry_price"}:
            return float(position.entry_price)
        if reference in {"signal_low", "entry_low"}:
            return float(df["low"].iloc[position.entry_idx])
        if reference in {"signal_close", "entry_close"}:
            return float(df["close"].iloc[position.entry_idx])

        if reference == "pre_event_close" and ctx is not None and ctx.pre_event_close is not None:
            return float(ctx.pre_event_close)

        if reference in df.columns:
            value = df[reference].iloc[min(idx, len(df) - 1)]
            return float(value) if pd.notna(value) else None

        return None

    @staticmethod
    def _get_sector(ctx: IndicatorContext | None) -> str | None:
        """Resolve the current symbol sector from context."""
        if ctx and ctx.stock_info and ctx.stock_info.sector:
            return ctx.stock_info.sector
        if ctx and ctx.sector_info and ctx.sector_info.name:
            return ctx.sector_info.name
        return None

    def _regime_allows_entry(
        self,
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> bool:
        """Check preferred market regime gating before entry."""
        if not strategy.meta.preferred_regime:
            return True

        regime = self._get_current_regime(df, idx, ctx)
        if regime is None:
            return True
        return regime.value in set(strategy.meta.preferred_regime)

    def _get_current_regime(
        self,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> MarketRegime | None:
        """Detect current regime from benchmark data (preferred) or symbol data."""
        benchmark_df = None
        if ctx and isinstance(ctx.benchmark_df, pd.DataFrame):
            benchmark_df = ctx.benchmark_df
        elif ctx and ctx.market_context and ctx.market_context.regime is not None:
            return ctx.market_context.regime

        source_df = benchmark_df if benchmark_df is not None else df
        cache_key = (id(source_df), idx)
        if cache_key in self._regime_cache:
            return self._regime_cache[cache_key]

        end_idx = min(idx + 1, len(source_df))
        start_idx = max(0, end_idx - 80)
        window = source_df.iloc[start_idx:end_idx].copy()

        if len(window) < 20:
            self._regime_cache[cache_key] = None
            return None

        for col in ("high", "low"):
            if col not in window.columns:
                window[col] = window["close"]

        detector = RegimeDetector()
        regime = detector.detect(window.reset_index(drop=True))
        self._regime_cache[cache_key] = regime
        return regime

    @staticmethod
    def _collect_unknown_indicators(strategy: Strategy) -> list[str]:
        """Collect indicator names not registered in IndicatorRegistry."""
        names = [c.indicator for c in strategy.entry.conditions]
        names.extend(c.indicator for c in strategy.entry.filters)
        if strategy.exit.stop_loss.conditions:
            names.extend(c.indicator for c in strategy.exit.stop_loss.conditions)

        unknown = {name for name in names if not IndicatorRegistry.is_registered(name)}
        return sorted(unknown)

    @staticmethod
    def _collect_all_indicators(strategy: Strategy) -> list[str]:
        """Collect all indicator names used by the strategy."""
        names = [c.indicator for c in strategy.entry.conditions]
        names.extend(c.indicator for c in strategy.entry.filters)
        if strategy.exit.stop_loss.conditions:
            names.extend(c.indicator for c in strategy.exit.stop_loss.conditions)
        return sorted(set(names))

    @staticmethod
    def _capture_indicators(
        strategy: Strategy,
        df: pd.DataFrame,
        idx: int,
        ctx: IndicatorContext | None,
    ) -> dict[str, float | bool]:
        """Capture all strategy-relevant indicator values at entry.

        This snapshot is attached to TradeSignal so the reflection layer
        can see exactly what indicators looked like when the trade was taken.
        """
        snapshot: dict[str, float | bool] = {}
        indicators_to_capture = {c.indicator for c in strategy.entry.conditions}
        indicators_to_capture.update(c.indicator for c in strategy.entry.filters)
        # Also capture core context indicators the LLM often references
        for extra in ("rsi_14", "volume_ratio_1d_5d", "atr", "momentum_10d", "ma20_slope"):
            indicators_to_capture.add(extra)
        for name in sorted(indicators_to_capture):
            if IndicatorRegistry.is_registered(name):
                try:
                    val = IndicatorRegistry.compute(name, df, idx, ctx)
                    if isinstance(val, bool):
                        snapshot[name] = val
                    else:
                        snapshot[name] = round(float(val), 4)
                except Exception:
                    pass
        return snapshot

    @staticmethod
    def _get_date(df: pd.DataFrame, idx: int) -> date:
        """Extract date from DataFrame row."""
        val = df.iloc[idx].get("date", None)
        if val is None:
            return date.today()
        if isinstance(val, date):
            return val
        try:
            parsed = pd.Timestamp(val).date()
            return parsed if isinstance(parsed, date) else date.today()
        except Exception:
            return date.today()


def _indicator_warmup(indicator_names: list[str]) -> int:
    """Estimate the minimum bar index required for all indicators to produce valid values.

    Parses period numbers from indicator names (e.g., ``ma60`` -> 60,
    ``rsi_14`` -> 14, ``bollinger_band_width_30d`` -> 30) and returns
    the maximum period found with a convergence buffer.

    EMA-based indicators (RSI, MACD, Bollinger) need approximately 2x period
    bars for the exponential weighting to stabilise. SMA-based indicators
    only need exactly *period* bars.  We use ``max_period * 2`` as a safe
    lower bound.
    """
    import re

    max_period = 0
    has_ema_indicator = False
    for name in indicator_names:
        # Extract all numbers that look like period lengths
        for m in re.finditer(r"(\d+)", name):
            val = int(m.group(1))
            # Heuristic: reasonable period range 2-500; ignore tiny numbers
            # that are more likely version suffixes (e.g., _v1)
            if 2 <= val <= 500:
                max_period = max(max_period, val)
        # Detect EMA-based indicators that need longer warmup
        lower = name.lower()
        if any(tag in lower for tag in ("rsi", "macd", "ema", "ewm", "bollinger")):
            has_ema_indicator = True

    # MACD needs slow + signal periods of warmup
    for name in indicator_names:
        if "macd" in name.lower():
            max_period = max(max_period, 35)  # 26 + 9 default
            has_ema_indicator = True

    # EMA convergence needs ~2x period; SMA needs exactly period
    buffer = max_period if has_ema_indicator else max(5, max_period // 4)
    return max(30, max_period + buffer)
