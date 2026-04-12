"""Data adapter abstract base class and DataManager."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

from alphaevo.data.cache import DataCache

if TYPE_CHECKING:
    from datetime import date

    from alphaevo.core.config import AppConfig
    from alphaevo.models.enums import MarketType
    from alphaevo.models.market import (
        EventContextSeries,
        MarketSnapshot,
        RealTimeQuote,
        SectorInfo,
        StockInfo,
    )


class DataAdapter(ABC):
    """Abstract interface for market data sources.

    Implement this to plug in any data provider (yfinance, akshare,
    daily_stock_analysis, etc.).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""

    @abstractmethod
    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV data.

        Returns DataFrame with columns:
        date, open, high, low, close, volume, amount (optional)
        """

    @abstractmethod
    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Get list of stocks in a market."""

    async def get_realtime_quote(self, symbol: str) -> RealTimeQuote | None:
        """Get real-time quote (optional, not all adapters support this)."""
        return None

    async def get_sector_data(self, symbol: str) -> SectorInfo | None:
        """Get sector/industry information (optional)."""
        return None

    async def get_snapshot(self, symbol: str, target_date: date) -> MarketSnapshot | None:
        """Build a full MarketSnapshot for a symbol on a given date.

        Default implementation fetches daily data and constructs a snapshot.
        Adapters may override for efficiency.
        """
        return None

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch benchmark index data.

        Adapters that do not support indices return an empty DataFrame.
        """
        return pd.DataFrame()

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch optional date-aligned event/news context for a symbol.

        Providers that do not support event/news context should return ``None``.
        The returned series is expected to align to trading dates, not just sparse
        event timestamps, so it can be injected directly into backtest data.
        """
        return None


class DataManager:
    """Unified data manager with multi-source fallback.

    Tries adapters in priority order. If the primary fails, falls
    through to the next available adapter.
    """

    def __init__(
        self,
        adapters: list[DataAdapter],
        *,
        cache: DataCache | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("At least one DataAdapter is required")
        self._adapters = adapters
        self._cache = cache

    @property
    def primary(self) -> DataAdapter:
        return self._adapters[0]

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily data with fallback across adapters."""
        errors: list[tuple[str, Exception]] = []
        for adapter in self._adapters:
            try:
                df = await adapter.get_daily_data(symbol, days)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                errors.append((adapter.name, e))
                continue
        detail = "; ".join(f"{name}: {err}" for name, err in errors)
        raise RuntimeError(f"All data adapters failed for {symbol}: {detail}")

    async def get_history(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch history for a date range."""
        if start > end:
            raise ValueError(f"start date ({start}) must be <= end date ({end})")
        cached = self._get_cached_history(symbol, start, end)
        if cached is not None:
            return cached

        days = (end - start).days + 30  # buffer for non-trading days
        df = await self.get_daily_data(symbol, days)
        history = self._prepare_history_frame(symbol, df, start=start, end=end)
        if history.empty:
            raise RuntimeError(f"No historical data returned for {symbol}: {start} → {end}")

        if self._cache is not None:
            self._cache.put(symbol, start, end, history)
        return history.copy()

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Get stock list with fallback."""
        for adapter in self._adapters:
            try:
                stocks = await adapter.get_stock_list(market)
                if stocks:
                    return stocks
            except Exception:
                continue
        return []

    async def get_snapshot(self, symbol: str, target_date: date) -> MarketSnapshot | None:
        """Get full snapshot with fallback."""
        for adapter in self._adapters:
            try:
                snapshot = await adapter.get_snapshot(symbol, target_date)
                if snapshot is not None:
                    return snapshot
            except Exception:
                continue
        return None

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        """Fetch benchmark index data with adapter fallback."""
        if start > end:
            raise ValueError(f"start date ({start}) must be <= end date ({end})")

        cache_key = f"__index__{index_symbol}"
        cached = self._get_cached_history(cache_key, start, end)
        if cached is not None:
            return cached

        for adapter in self._adapters:
            try:
                df = await adapter.get_index_data(index_symbol, start, end)
                if df is not None and not df.empty:
                    history = self._prepare_history_frame(
                        index_symbol,
                        df,
                        start=start,
                        end=end,
                        required_cols={"date", "close"},
                    )
                    if history.empty:
                        continue
                    if self._cache is not None:
                        self._cache.put(cache_key, start, end, history)
                    return history.copy()
            except Exception:
                continue
        return pd.DataFrame()

    async def get_sector_data(self, symbol: str) -> SectorInfo | None:
        """Fetch sector information with adapter fallback."""
        for adapter in self._adapters:
            try:
                info = await adapter.get_sector_data(symbol)
                if info is not None:
                    return info
            except Exception:
                continue
        return None

    async def get_event_context(
        self,
        symbol: str,
        start: date,
        end: date,
    ) -> EventContextSeries | None:
        """Fetch event/news context with adapter fallback."""
        for adapter in self._adapters:
            try:
                context = await adapter.get_event_context(symbol, start, end)
                if context is not None and context.records:
                    return context
            except Exception:
                continue
        return None

    def _get_cached_history(
        self,
        cache_key: str,
        start: date,
        end: date,
    ) -> pd.DataFrame | None:
        """Return a normalized cached history frame when available."""
        if self._cache is None:
            return None

        cached = self._cache.get(cache_key, start, end)
        if cached is None or cached.empty:
            return None
        return self._prepare_history_frame(cache_key, cached, start=start, end=end)

    @staticmethod
    def _prepare_history_frame(
        symbol: str,
        df: pd.DataFrame,
        *,
        start: date,
        end: date,
        required_cols: set[str] | None = None,
    ) -> pd.DataFrame:
        """Normalize and clip a history dataframe to the requested date range."""
        required = required_cols or {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Adapter returned DataFrame missing required columns for {symbol}: {missing}"
            )

        normalized = df.copy()
        normalized["date"] = pd.to_datetime(normalized["date"]).dt.date
        normalized = normalized.sort_values("date").drop_duplicates(subset=["date"], keep="last")

        # Recompute prev_close BEFORE clipping so the first row in the
        # clipped range still has a valid value (not NaN).
        if "prev_close" not in normalized.columns or normalized["prev_close"].isna().any():
            normalized["prev_close"] = normalized["close"].shift(1)

        mask = (normalized["date"] >= start) & (normalized["date"] <= end)
        return normalized.loc[mask].reset_index(drop=True)


def get_adapter(config: AppConfig) -> DataAdapter:
    """Instantiate the configured primary adapter for lightweight workflows."""
    adapter_name = config.data.adapter

    if adapter_name == "yfinance":
        from alphaevo.data.adapters.yfinance import YFinanceAdapter

        return YFinanceAdapter()

    if adapter_name == "akshare":
        from alphaevo.data.adapters.akshare import AkShareAdapter

        return AkShareAdapter()

    if adapter_name == "dsa":
        from alphaevo.data.adapters.dsa import DSAAdapter

        try:
            return DSAAdapter(dsa_path=config.data.dsa_path)
        except ImportError as err:
            raise ValueError(
                "Adapter 'dsa' is an optional daily_stock_analysis bridge. "
                "Configure ALPHAEVO_DSA_PATH (or data.dsa_path) to enable it."
            ) from err

    raise ValueError(f"Unknown adapter: {adapter_name}")
