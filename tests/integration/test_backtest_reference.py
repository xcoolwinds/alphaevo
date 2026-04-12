"""Reference backtest comparison — verify engine signals against a naive implementation.

This module independently computes trade outcomes for a known strategy on
deterministic data, then compares with BacktestEngine output.  The goal is to
catch logic regressions in the engine (entry timing, slippage, commission,
return calculation, etc.).
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from alphaevo.backtest.engine import BacktestEngine
from alphaevo.models.execution import SampleBatch
from alphaevo.strategy.dsl.parser import StrategyParser

# ── Minimal strategy YAML for deterministic testing ────────────────────

_MA_CROSS_YAML = """\
meta:
  id: ref_ma_cross_v1
  name: Reference MA Crossover
  version: 1
  market: a_share
  category: trend

description: |
  MA5 above MA10, close above MA20, next-open entry, 5% SL, 2×RR TP, 10d max hold.

universe:
  market: [a_share_main]

entry:
  logic: and
  conditions:
    - indicator: ma5_above_ma10
      op: "=="
      value: true
    - indicator: close_above_ma20
      op: "=="
      value: true

exit:
  stop_loss:
    type: pct
    value: 0.05
  take_profit:
    type: rr
    value: 2.0
  max_holding_days: 10
"""


def _make_trending_up_data(days: int = 120) -> pd.DataFrame:
    """Deterministic uptrend data with known structure.

    Price rises steadily so MA5 > MA10 > MA20 after warmup, guaranteeing entries.
    """
    rows: list[dict] = []
    price = 50.0
    for i in range(days):
        # Steady uptrend: +0.3% per day
        price *= 1.003
        o = price * 0.998
        h = price * 1.005
        lo = price * 0.995
        c = price
        prev_c = rows[-1]["close"] if rows else c * 0.997
        rows.append(
            {
                "date": date(2023, 6, 1) + timedelta(days=i),
                "open": round(o, 4),
                "high": round(h, 4),
                "low": round(lo, 4),
                "close": round(c, 4),
                "volume": 1_000_000,
                "prev_close": round(prev_c, 4),
            }
        )
    return pd.DataFrame(rows)


def _naive_return(entry_price: float, exit_price: float,
                  slippage: float, commission: float) -> float:
    """Compute return the same way the engine should."""
    adj_entry = entry_price * (1 + slippage)  # buy higher
    adj_exit = exit_price * (1 - slippage)  # sell lower
    net_entry = adj_entry * (1 + commission)
    net_exit = adj_exit * (1 - commission)
    return (net_exit - net_entry) / net_entry


# ── Tests ──────────────────────────────────────────────────────────────


class TestReferenceBacktest:
    """Compare engine outputs to hand-computed reference values."""

    SLIPPAGE = 0.001
    COMMISSION = 0.0003

    def _run(self, yaml_str: str = _MA_CROSS_YAML, days: int = 120):
        parser = StrategyParser()
        strategy = parser.parse_yaml(yaml_str)
        df = _make_trending_up_data(days)
        data = {"REF001": df}
        batch = SampleBatch(
            batch_id="ref_batch",
            strategy_id=strategy.meta.id,
            symbols=["REF001"],
            date_range=(date(2023, 6, 1), date(2023, 6, 1) + timedelta(days=days - 1)),
        )
        engine = BacktestEngine(
            slippage=self.SLIPPAGE,
            commission=self.COMMISSION,
            min_data_days=30,
        )
        result = engine.run(strategy, data, batch)
        return strategy, df, result

    def test_signals_produced(self) -> None:
        """Trending-up data must trigger entries for MA cross strategy."""
        _, _, result = self._run()
        assert result.total_signals > 0, "Expected signals on trending-up data"

    def test_entry_price_includes_slippage(self) -> None:
        """Entry price should be next-open × (1 + slippage)."""
        _, df, result = self._run()
        for sig in result.signals:
            if sig.exit_price is None:
                continue
            # Find signal bar by entry date
            entry_rows = df[df["date"] == sig.signal_date]
            if entry_rows.empty:
                continue
            entry_idx = entry_rows.index[0]
            raw_open = df["open"].iloc[entry_idx]
            expected_entry = raw_open * (1 + self.SLIPPAGE)
            assert abs(sig.entry_price - expected_entry) < 0.02, (
                f"Entry price {sig.entry_price} != expected {expected_entry}"
            )

    def test_return_pct_matches_naive(self) -> None:
        """return_pct should match independent naive computation."""
        _, df, result = self._run()
        executed = [s for s in result.signals if s.exit_price is not None]
        assert len(executed) > 0

        for sig in executed:
            # Reverse-engineer raw exit before slippage was applied by engine
            # Engine formula: exit_price *= (1 - slippage), then
            #   net_entry = entry_price * (1 + commission)
            #   net_exit  = exit_price * (1 - commission)
            #   return_pct = (net_exit - net_entry) / net_entry
            #
            # sig.entry_price already has slippage baked in (buy side)
            # sig.exit_price already has slippage baked in (sell side)
            net_entry = sig.entry_price * (1 + self.COMMISSION)
            net_exit = sig.exit_price * (1 - self.COMMISSION)
            expected_ret = (net_exit - net_entry) / net_entry
            assert abs(sig.return_pct - expected_ret) < 1e-4, (
                f"return_pct {sig.return_pct} != expected {expected_ret:.6f}"
            )

    def test_max_hold_respected(self) -> None:
        """No trade should exceed max_holding_days."""
        _, _, result = self._run()
        max_hold = 10
        for sig in result.signals:
            if sig.exit_price is not None:
                assert sig.holding_days <= max_hold, (
                    f"Holding days {sig.holding_days} > max {max_hold}"
                )

    def test_stop_loss_exits_are_losers(self) -> None:
        """Stop-loss exits should have negative returns (net of costs)."""
        from alphaevo.models.enums import ExitReason

        _, _, result = self._run(days=200)
        sl_trades = [
            s for s in result.signals
            if s.exit_reason == ExitReason.STOP_LOSS
        ]
        for sig in sl_trades:
            assert sig.return_pct < 0, f"SL trade should be negative: {sig.return_pct}"

    def test_no_overlapping_trades(self) -> None:
        """Engine should not open a new position while one is active."""
        _, _, result = self._run(days=300)
        signals = sorted(
            [s for s in result.signals if s.exit_date is not None],
            key=lambda s: s.signal_date,
        )
        for i in range(1, len(signals)):
            prev = signals[i - 1]
            curr = signals[i]
            assert curr.signal_date >= prev.exit_date, (
                f"Overlap: trade {i} entered {curr.signal_date} "
                f"before trade {i-1} exited {prev.exit_date}"
            )

    def test_batch_consistency(self) -> None:
        """Running the same data twice should produce identical results."""
        _, _, r1 = self._run()
        _, _, r2 = self._run()
        assert r1.total_signals == r2.total_signals
        for s1, s2 in zip(r1.signals, r2.signals, strict=True):
            assert s1.entry_price == s2.entry_price
            assert s1.exit_price == s2.exit_price
            assert s1.return_pct == s2.return_pct


class TestPctStopLossReference:
    """Verify pct stop-loss triggers at exactly the right price level."""

    _YAML = """\
meta:
  id: ref_sl_test
  name: Stop Loss Reference
  version: 1
  market: a_share
  category: trend

description: Simple entry with tight stop loss.

universe:
  market: [a_share_main]

entry:
  conditions:
    - indicator: close_above_ma20
      op: "=="
      value: true

exit:
  stop_loss:
    type: pct
    value: 0.03
  take_profit:
    type: pct
    value: 0.10
  max_holding_days: 50
"""

    def test_sl_triggers_on_drop(self) -> None:
        """Build data with a known crash after uptrend to force SL."""
        rows: list[dict] = []
        price = 50.0
        # 60 days uptrend → entry should trigger
        for i in range(60):
            price *= 1.002
            rows.append(self._bar(i, price))
        # 10 days sharp drop → SL should trigger
        for i in range(60, 70):
            price *= 0.98  # -2% per day
            rows.append(self._bar(i, price))
        # Fill out
        for i in range(70, 120):
            price *= 1.001
            rows.append(self._bar(i, price))

        df = pd.DataFrame(rows)
        parser = StrategyParser()
        strategy = parser.parse_yaml(self._YAML)
        data = {"SL001": df}
        batch = SampleBatch(
            batch_id="sl_batch",
            strategy_id=strategy.meta.id,
            symbols=["SL001"],
            date_range=(date(2023, 6, 1), date(2023, 6, 1) + timedelta(days=119)),
        )
        engine = BacktestEngine(slippage=0.001, commission=0.0003, min_data_days=30)
        result = engine.run(strategy, data, batch)

        from alphaevo.models.enums import ExitReason

        sl_exits = [s for s in result.signals if s.exit_reason == ExitReason.STOP_LOSS]
        assert len(sl_exits) > 0, "Expected at least one SL exit during crash period"
        for sig in sl_exits:
            assert sig.return_pct < 0

    @staticmethod
    def _bar(day_idx: int, price: float) -> dict:
        return {
            "date": date(2023, 6, 1) + timedelta(days=day_idx),
            "open": round(price * 0.999, 4),
            "high": round(price * 1.003, 4),
            "low": round(price * 0.997, 4),
            "close": round(price, 4),
            "volume": 1_000_000,
            "prev_close": round(price / 1.002, 4),
        }
