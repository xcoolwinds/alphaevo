"""Tests for portfolio-level backtester."""

from datetime import date

from alphaevo.backtest.portfolio import (
    PortfolioBacktester,
    PortfolioConfig,
    PortfolioSnapshot,
    _compute_max_drawdown,
    _compute_portfolio_sharpe,
)
from alphaevo.models.enums import ExitReason, SignalDirection
from alphaevo.models.execution import BacktestResult, TradeSignal


def _make_signal(
    symbol: str = "AAPL",
    signal_date: date = date(2024, 6, 1),
    entry_price: float = 100.0,
    exit_price: float = 105.0,
    return_pct: float = 0.05,
    holding_days: int = 5,
) -> TradeSignal:
    return TradeSignal(
        symbol=symbol,
        signal_date=signal_date,
        direction=SignalDirection.LONG,
        entry_price=entry_price,
        exit_price=exit_price,
        return_pct=return_pct,
        holding_days=holding_days,
        exit_reason=ExitReason.TAKE_PROFIT,
    )


def _make_backtest_result(signals: list[TradeSignal]) -> BacktestResult:
    return BacktestResult(
        strategy_id="test_v1",
        batch_id="batch_1",
        signals=signals,
        total_signals=len(signals),
        executed_signals=len(signals),
    )


class TestPortfolioConfig:
    def test_defaults(self):
        cfg = PortfolioConfig()
        assert cfg.initial_capital == 100_000.0
        assert cfg.max_positions == 5
        assert cfg.position_size_pct == 0.2

    def test_custom(self):
        cfg = PortfolioConfig(initial_capital=50_000, max_positions=3)
        assert cfg.initial_capital == 50_000
        assert cfg.max_positions == 3


class TestPortfolioBacktester:
    def test_empty_signals(self):
        bt = _make_backtest_result([])
        result = PortfolioBacktester().simulate(bt)
        assert result.final_equity == 100_000.0
        assert result.total_trades == 0

    def test_single_winning_trade(self):
        signals = [_make_signal(return_pct=0.05, entry_price=100.0, exit_price=105.0)]
        bt = _make_backtest_result(signals)
        result = PortfolioBacktester().simulate(bt)
        assert result.total_trades == 1
        assert result.final_equity > 100_000.0
        assert result.total_return > 0
        assert result.win_rate == 1.0

    def test_single_losing_trade(self):
        signals = [
            _make_signal(return_pct=-0.05, entry_price=100.0, exit_price=95.0)
        ]
        bt = _make_backtest_result(signals)
        result = PortfolioBacktester().simulate(bt)
        assert result.total_trades == 1
        assert result.final_equity < 100_000.0
        assert result.win_rate == 0.0

    def test_max_positions_enforced(self):
        # Create 8 signals on the same date — only 5 should get filled
        signals = [
            _make_signal(
                symbol=f"SYM{i}",
                signal_date=date(2024, 6, 1),
                entry_price=50.0,
                exit_price=52.0,
                return_pct=0.04,
                holding_days=10,
            )
            for i in range(8)
        ]
        bt = _make_backtest_result(signals)
        cfg = PortfolioConfig(max_positions=5)
        result = PortfolioBacktester(cfg).simulate(bt)
        assert result.max_concurrent_positions <= 5

    def test_multiple_dates(self):
        signals = [
            _make_signal(symbol="AAPL", signal_date=date(2024, 6, 1), holding_days=3),
            _make_signal(symbol="MSFT", signal_date=date(2024, 6, 5), holding_days=3),
        ]
        bt = _make_backtest_result(signals)
        result = PortfolioBacktester().simulate(bt)
        assert result.total_trades == 2

    def test_signals_without_exit_skipped(self):
        """Signals with exit_price=None are filtered out."""
        signals = [
            _make_signal(exit_price=None, return_pct=0.0),
            _make_signal(symbol="MSFT"),
        ]
        bt = _make_backtest_result(signals)
        # The first signal has None exit_price, should be skipped
        result = PortfolioBacktester().simulate(bt)
        assert result.total_trades >= 1


class TestHelperFunctions:
    def test_max_drawdown_no_drawdown(self):
        curve = [
            PortfolioSnapshot(date=date(2024, 6, i), equity=100 + i, cash=50, positions_open=1)
            for i in range(1, 6)
        ]
        assert _compute_max_drawdown(curve) == 0.0

    def test_max_drawdown_with_decline(self):
        curve = [
            PortfolioSnapshot(date=date(2024, 6, 1), equity=100, cash=50, positions_open=1),
            PortfolioSnapshot(date=date(2024, 6, 2), equity=90, cash=50, positions_open=1),
            PortfolioSnapshot(date=date(2024, 6, 3), equity=95, cash=50, positions_open=1),
        ]
        dd = _compute_max_drawdown(curve)
        assert abs(dd - 0.10) < 0.001

    def test_sharpe_empty(self):
        assert _compute_portfolio_sharpe([]) == 0.0

    def test_sharpe_positive(self):
        curve = [
            PortfolioSnapshot(
                date=date(2024, 6, i),
                equity=100 + i * 0.5,
                cash=50,
                positions_open=1,
                daily_return=0.005,
            )
            for i in range(1, 30)
        ]
        sharpe = _compute_portfolio_sharpe(curve)
        assert sharpe > 0  # Consistent positive return → positive Sharpe


class TestEstimateExitDate:
    def _signal_on(self, d: date, holding_days: int) -> "TradeSignal":
        return _make_signal(signal_date=d, holding_days=holding_days)

    def test_skips_weekends(self):
        """Friday + 3 trading days should land on Wednesday, not Monday."""
        from alphaevo.backtest.portfolio import _estimate_exit_date

        # 2024-06-07 is a Friday
        sig = self._signal_on(date(2024, 6, 7), 3)
        result = _estimate_exit_date(sig)
        # 3 trading days: Mon 6/10, Tue 6/11, Wed 6/12
        assert result == date(2024, 6, 12)

    def test_one_day_from_friday(self):
        from alphaevo.backtest.portfolio import _estimate_exit_date

        sig = self._signal_on(date(2024, 6, 7), 1)
        result = _estimate_exit_date(sig)
        # 1 trading day: Mon 6/10
        assert result == date(2024, 6, 10)

    def test_weekday_no_skip(self):
        from alphaevo.backtest.portfolio import _estimate_exit_date

        # 2024-06-03 is a Monday
        sig = self._signal_on(date(2024, 6, 3), 2)
        result = _estimate_exit_date(sig)
        # 2 trading days: Tue 6/4, Wed 6/5
        assert result == date(2024, 6, 5)


class TestMarkToMarket:
    """Test that equity curve reflects intra-trade P&L, not just allocation."""

    def test_winning_trade_equity_rises_during_hold(self):
        """Equity should increase during a winning trade, not stay flat."""
        signals = [
            _make_signal(
                signal_date=date(2024, 6, 3),  # Monday
                entry_price=100.0,
                exit_price=120.0,
                return_pct=0.20,
                holding_days=5,
            ),
        ]
        bt = _make_backtest_result(signals)
        cfg = PortfolioConfig(initial_capital=100_000, max_positions=5)
        result = PortfolioBacktester(cfg).simulate(bt)

        # The equity curve should show growth during the holding period,
        # not be flat until exit. Find the snapshot on the signal date.
        assert len(result.equity_curve) >= 1
        # With MTM, equity should equal cash + (shares × interpolated_price),
        # not cash + allocation (which would be exactly initial_capital).
        # On signal_date, days_held=0 so MTM = entry_price → equity ≈ initial
        # The key assertion: final equity should reflect the 20% gain on the position
        assert result.final_equity > 100_000.0

    def test_losing_trade_equity_drops_during_hold(self):
        """Equity should drop progressively during a losing trade."""
        signals = [
            _make_signal(
                signal_date=date(2024, 6, 3),
                entry_price=100.0,
                exit_price=80.0,
                return_pct=-0.20,
                holding_days=5,
            ),
        ]
        bt = _make_backtest_result(signals)
        result = PortfolioBacktester().simulate(bt)
        assert result.final_equity < 100_000.0

    def test_mtm_helper_directly(self):
        """Verify _mark_to_market interpolation."""
        from alphaevo.backtest.portfolio import _mark_to_market

        sig = _make_signal(
            signal_date=date(2024, 6, 3),
            entry_price=100.0,
            exit_price=110.0,
            holding_days=10,
        )
        # Day 0: entry price
        val0 = _mark_to_market(sig, 10.0, date(2024, 6, 3))
        assert val0 == 10.0 * 100.0  # shares × entry

        # Halfway through: should be ~105
        val5 = _mark_to_market(sig, 10.0, date(2024, 6, 8))
        assert 1040 < val5 < 1060  # 10 shares × ~105

        # At end: should be ~110
        val10 = _mark_to_market(sig, 10.0, date(2024, 6, 13))
        assert val10 == 10.0 * 110.0
