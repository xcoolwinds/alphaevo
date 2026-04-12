"""Tests for UniverseProvider implementations."""

from __future__ import annotations

import asyncio

import pytest

from alphaevo.data.universe import (
    AdapterUniverseProvider,
    CuratedUniverseProvider,
    CustomUniverseProvider,
    UniverseProvider,
)
from alphaevo.models.enums import MarketType


class TestCuratedUniverseProvider:
    def test_is_curated(self) -> None:
        provider = CuratedUniverseProvider()
        assert provider.is_curated is True
        assert provider.name == "curated"

    def test_get_stock_list_a_share(self) -> None:
        provider = CuratedUniverseProvider()
        stocks = asyncio.run(provider.get_stock_list(MarketType.A_SHARE))
        assert len(stocks) > 0
        assert all(s.market == MarketType.A_SHARE for s in stocks)

    def test_get_stock_list_us(self) -> None:
        provider = CuratedUniverseProvider()
        stocks = asyncio.run(provider.get_stock_list(MarketType.US))
        assert len(stocks) > 0
        symbols = {s.symbol for s in stocks}
        assert "AAPL" in symbols

    def test_get_stock_list_hk(self) -> None:
        provider = CuratedUniverseProvider()
        stocks = asyncio.run(provider.get_stock_list(MarketType.HK))
        assert len(stocks) > 0


class TestCustomUniverseProvider:
    def test_matching_market(self) -> None:
        provider = CustomUniverseProvider(["AAPL", "MSFT"], market=MarketType.US)
        stocks = asyncio.run(provider.get_stock_list(MarketType.US))
        assert len(stocks) == 2
        assert {s.symbol for s in stocks} == {"AAPL", "MSFT"}

    def test_non_matching_market(self) -> None:
        provider = CustomUniverseProvider(["AAPL"], market=MarketType.US)
        stocks = asyncio.run(provider.get_stock_list(MarketType.A_SHARE))
        assert len(stocks) == 0

    def test_is_not_curated(self) -> None:
        provider = CustomUniverseProvider(["AAPL"])
        assert provider.is_curated is False

    def test_name(self) -> None:
        provider = CustomUniverseProvider(["AAPL", "MSFT", "GOOGL"])
        assert "3 symbols" in provider.name


class TestAdapterUniverseProvider:
    def test_delegates_to_adapter(self) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from alphaevo.models.market import StockInfo

        adapter = MagicMock()
        adapter.name = "mock"
        mock_stocks = [
            StockInfo(symbol="TEST", name="Test", market=MarketType.US),
        ]
        adapter.get_stock_list = AsyncMock(return_value=mock_stocks)

        provider = AdapterUniverseProvider(adapter)
        assert provider.is_curated is False
        assert "mock" in provider.name

        stocks = asyncio.run(provider.get_stock_list(MarketType.US))
        assert stocks == mock_stocks
        adapter.get_stock_list.assert_awaited_once_with(MarketType.US)


class TestUniverseProviderABC:
    def test_cannot_instantiate_abc(self) -> None:
        with pytest.raises(TypeError):
            UniverseProvider()  # type: ignore[abstract]
