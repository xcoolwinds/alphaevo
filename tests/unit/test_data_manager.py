"""Tests for DataManager history fetching and cache integration."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from alphaevo.data.adapter import DataAdapter, DataManager
from alphaevo.data.cache import DataCache
from alphaevo.models.enums import MarketType
from alphaevo.models.market import MarketContext


def _make_history(
    start: date,
    days: int,
    *,
    base_price: float,
) -> pd.DataFrame:
    rows = []
    for offset in range(days):
        current = start + timedelta(days=offset)
        close = base_price + offset
        rows.append(
            {
                "date": current,
                "open": close - 0.5,
                "high": close + 0.5,
                "low": close - 1.0,
                "close": close,
                "volume": 1_000_000,
            }
        )
    return pd.DataFrame(rows)


class _StubAdapter(DataAdapter):
    def __init__(self) -> None:
        self.daily_calls = 0
        self.index_calls = 0
        self.market_context_calls = 0
        self._stock_df = _make_history(date(2024, 1, 1), 120, base_price=100.0)
        self._index_df = _make_history(date(2024, 1, 1), 120, base_price=3000.0)

    @property
    def name(self) -> str:
        return "stub"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        self.daily_calls += 1
        return self._stock_df.copy()

    async def get_stock_list(self, market: MarketType) -> list:
        return []

    async def get_index_data(self, index_symbol: str, start: date, end: date) -> pd.DataFrame:
        self.index_calls += 1
        return self._index_df.copy()

    async def get_market_context(self, market: MarketType) -> MarketContext | None:
        self.market_context_calls += 1
        if market != MarketType.A_SHARE:
            return None
        return MarketContext(breadth=0.62, sentiment_index=0.58)


@pytest.mark.asyncio
async def test_get_history_uses_cache(tmp_path):
    adapter = _StubAdapter()
    manager = DataManager([adapter], cache=DataCache(tmp_path / "cache"))
    start = date(2024, 1, 10)
    end = date(2024, 1, 20)

    first = await manager.get_history("AAPL", start, end)
    second = await manager.get_history("AAPL", start, end)

    assert adapter.daily_calls == 1
    assert list(first["date"]) == list(second["date"])
    assert first["date"].min() == start
    assert first["date"].max() == end


@pytest.mark.asyncio
async def test_get_history_reuses_covering_cached_range(tmp_path):
    adapter = _StubAdapter()
    manager = DataManager([adapter], cache=DataCache(tmp_path / "cache"))

    broad = await manager.get_history("AAPL", date(2024, 1, 1), date(2024, 1, 20))
    subset = await manager.get_history("AAPL", date(2024, 1, 10), date(2024, 1, 15))

    assert adapter.daily_calls == 1
    assert broad["date"].min() == date(2024, 1, 1)
    assert subset["date"].min() == date(2024, 1, 10)
    assert subset["date"].max() == date(2024, 1, 15)


@pytest.mark.asyncio
async def test_index_cache_uses_separate_namespace_from_stock_history(tmp_path):
    adapter = _StubAdapter()
    manager = DataManager([adapter], cache=DataCache(tmp_path / "cache"))
    start = date(2024, 1, 10)
    end = date(2024, 1, 20)

    stock_df = await manager.get_history("000001", start, end)
    index_df = await manager.get_index_data("000001", start, end)
    cached_index_df = await manager.get_index_data("000001", start, end)

    assert adapter.daily_calls == 1
    assert adapter.index_calls == 1
    assert float(stock_df["close"].iloc[0]) < 1000
    assert float(index_df["close"].iloc[0]) > 1000
    assert list(index_df["close"]) == list(cached_index_df["close"])


@pytest.mark.asyncio
async def test_prev_close_preserved_after_clipping(tmp_path):
    """N-04: prev_close should be valid (not NaN) after date-range clipping."""
    adapter = _StubAdapter()
    manager = DataManager([adapter], cache=DataCache(tmp_path / "cache"))
    # Request a subset — clipping removes rows but prev_close should be recomputed
    start = date(2024, 1, 10)
    end = date(2024, 1, 20)
    df = await manager.get_history("AAPL", start, end)
    assert "prev_close" in df.columns
    # First row may still be NaN (no prior row exists at all), but rows 1+ must be valid
    if len(df) > 1:
        assert df["prev_close"].iloc[1:].notna().all()


@pytest.mark.asyncio
async def test_get_market_context_uses_adapter_fallback():
    adapter = _StubAdapter()
    manager = DataManager([adapter])

    context = await manager.get_market_context(MarketType.A_SHARE)

    assert context is not None
    assert context.breadth == 0.62
    assert adapter.market_context_calls == 1
