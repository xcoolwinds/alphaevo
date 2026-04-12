"""Portfolio-level backtester — wraps signal-level engine with position sizing.

Adds:
- Initial capital and per-trade allocation
- Maximum concurrent positions
- Portfolio equity curve and drawdown
- Portfolio-level Sharpe ratio
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field

from alphaevo.models.execution import BacktestResult, TradeSignal


class PortfolioConfig(BaseModel):
    """Configuration for portfolio-level backtesting."""

    initial_capital: float = Field(default=100_000.0, gt=0)
    max_positions: int = Field(default=5, ge=1)
    position_size_pct: float = Field(default=0.2, gt=0, le=1.0)
    risk_per_trade_pct: float = Field(default=0.02, gt=0, le=0.5)


@dataclass
class PortfolioTrade:
    """A trade with portfolio-level sizing applied."""

    signal: TradeSignal
    shares: float
    capital_allocated: float
    pnl: float
    pnl_pct: float


@dataclass
class PortfolioSnapshot:
    """Daily portfolio snapshot for equity curve."""

    date: date
    equity: float
    cash: float
    positions_open: int
    daily_return: float = 0.0


@dataclass
class PortfolioResult:
    """Portfolio-level backtest result."""

    config: PortfolioConfig
    trades: list[PortfolioTrade] = field(default_factory=list)
    equity_curve: list[PortfolioSnapshot] = field(default_factory=list)
    final_equity: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    max_concurrent_positions: int = 0
    total_trades: int = 0
    capital_utilization: float = 0.0

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.pnl > 0)
        return winners / len(self.trades)


class PortfolioBacktester:
    """Wraps signal-level BacktestResult with portfolio-level position sizing.

    Takes the output of BacktestEngine.run() and simulates portfolio-level
    execution with capital constraints and position limits.
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def simulate(self, backtest_result: BacktestResult) -> PortfolioResult:
        """Run portfolio simulation on signal-level backtest results.

        Processes signals chronologically, applying position sizing and
        capital constraints. Signals that exceed the max_positions limit
        or available capital are skipped.
        """
        cfg = self.config
        cash = cfg.initial_capital
        equity = cfg.initial_capital

        # Sort signals by entry date
        signals = sorted(
            [s for s in backtest_result.signals if s.exit_price is not None],
            key=lambda s: s.signal_date,
        )

        if not signals:
            return PortfolioResult(
                config=cfg,
                final_equity=cfg.initial_capital,
            )

        portfolio_trades: list[PortfolioTrade] = []
        open_positions: list[tuple[TradeSignal, float, float]] = []  # (signal, shares, allocated)
        equity_curve: list[PortfolioSnapshot] = []
        max_concurrent = 0
        total_capital_days = 0.0
        total_days = 0

        # Group signals by date for processing
        all_dates = sorted(set(s.signal_date for s in signals))
        signal_idx = 0

        for current_date in all_dates:
            # Close positions that have exited
            still_open: list[tuple[TradeSignal, float, float]] = []
            for sig, shares, allocated in open_positions:
                exit_date = _estimate_exit_date(sig)
                if exit_date is not None and exit_date <= current_date:
                    # Close this position
                    pnl = shares * (sig.exit_price - sig.entry_price) if sig.exit_price else 0.0
                    pnl_pct = sig.return_pct
                    cash += allocated + pnl
                    portfolio_trades.append(
                        PortfolioTrade(
                            signal=sig,
                            shares=shares,
                            capital_allocated=allocated,
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                        )
                    )
                else:
                    still_open.append((sig, shares, allocated))
            open_positions = still_open

            # Open new positions from today's signals
            while signal_idx < len(signals) and signals[signal_idx].signal_date == current_date:
                sig = signals[signal_idx]
                signal_idx += 1

                if len(open_positions) >= cfg.max_positions:
                    continue  # Skip — position limit reached

                allocation = min(
                    cash * cfg.position_size_pct,
                    equity * cfg.position_size_pct,
                )
                if allocation < sig.entry_price:
                    continue  # Not enough capital

                shares = allocation / sig.entry_price
                cash -= allocation
                open_positions.append((sig, shares, allocation))

            max_concurrent = max(max_concurrent, len(open_positions))

            # Calculate current equity using mark-to-market estimates.
            # Since we only have entry/exit prices (not daily closes), we
            # linearly interpolate each position's value between entry and
            # exit price over the holding period.
            invested_mtm = 0.0
            for sig, shares, _alloc in open_positions:
                invested_mtm += _mark_to_market(sig, shares, current_date)
            current_equity = cash + invested_mtm
            prev_equity = equity_curve[-1].equity if equity_curve else cfg.initial_capital
            daily_ret = (current_equity / prev_equity - 1) if prev_equity > 0 else 0.0

            equity_curve.append(
                PortfolioSnapshot(
                    date=current_date,
                    equity=current_equity,
                    cash=cash,
                    positions_open=len(open_positions),
                    daily_return=daily_ret,
                )
            )

            total_capital_days += len(open_positions) * cfg.position_size_pct
            total_days += 1

        # Close any remaining positions at their exit prices
        for sig, shares, allocated in open_positions:
            pnl = shares * (sig.exit_price - sig.entry_price) if sig.exit_price else 0.0
            cash += allocated + pnl
            portfolio_trades.append(
                PortfolioTrade(
                    signal=sig,
                    shares=shares,
                    capital_allocated=allocated,
                    pnl=pnl,
                    pnl_pct=sig.return_pct,
                )
            )

        final_equity = cash
        total_return = (final_equity / cfg.initial_capital - 1) if cfg.initial_capital > 0 else 0.0
        max_dd = _compute_max_drawdown(equity_curve)
        sharpe = _compute_portfolio_sharpe(equity_curve)
        cap_util = (total_capital_days / (total_days * cfg.max_positions)) if total_days > 0 else 0.0

        return PortfolioResult(
            config=cfg,
            trades=portfolio_trades,
            equity_curve=equity_curve,
            final_equity=final_equity,
            total_return=total_return,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            max_concurrent_positions=max_concurrent,
            total_trades=len(portfolio_trades),
            capital_utilization=min(1.0, cap_util),
        )


def _mark_to_market(signal: TradeSignal, shares: float, current_date: date) -> float:
    """Estimate position value via linear interpolation between entry and exit.

    Since we only have signal-level entry/exit prices (not daily OHLCV),
    we linearly interpolate the price over the holding period to produce
    a realistic equity curve with intra-trade P&L visibility.
    """
    if signal.holding_days <= 0 or signal.exit_price is None:
        return shares * signal.entry_price  # Fallback to entry value

    days_held = (current_date - signal.signal_date).days
    if days_held <= 0:
        return shares * signal.entry_price

    progress = min(1.0, days_held / max(1, signal.holding_days))
    estimated_price = signal.entry_price + progress * (signal.exit_price - signal.entry_price)
    return shares * estimated_price


def _estimate_exit_date(signal: TradeSignal) -> date | None:
    """Estimate exit date from signal_date + holding_days (trading days).

    Skips weekends (Saturday/Sunday) to approximate real market schedules.
    Does not handle exchange-specific holidays.
    """
    if signal.holding_days <= 0:
        return None

    from datetime import timedelta

    current = signal.signal_date
    remaining = signal.holding_days
    while remaining > 0:
        current += timedelta(days=1)
        # Skip weekends: 5=Saturday, 6=Sunday
        if current.weekday() < 5:
            remaining -= 1
    return current


def _compute_max_drawdown(curve: list[PortfolioSnapshot]) -> float:
    """Compute maximum drawdown from equity curve."""
    if not curve:
        return 0.0
    peak = curve[0].equity
    max_dd = 0.0
    for snap in curve:
        if snap.equity > peak:
            peak = snap.equity
        dd = (peak - snap.equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _compute_portfolio_sharpe(
    curve: list[PortfolioSnapshot],
    risk_free_rate: float = 0.02,
) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(curve) < 2:
        return 0.0
    returns = [snap.daily_return for snap in curve if snap.daily_return != 0.0]
    if len(returns) < 2:
        return 0.0
    avg = sum(returns) / len(returns)
    var = sum((r - avg) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return 0.0
    daily_rf = risk_free_rate / 252
    return (avg - daily_rf) / std * math.sqrt(252)
