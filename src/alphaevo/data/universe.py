"""Universe providers — define stock pools for strategy research.

Separates the concept of "which stocks to research" from the data adapter.
This avoids the default curated list being misinterpreted as full market
coverage.

Usage::

    # Demo / quickstart — curated list of liquid stocks
    provider = CuratedUniverseProvider()

    # Research — dynamically fetch all stocks from data adapter
    provider = AdapterUniverseProvider(adapter)

    # Custom — user-supplied ticker list
    provider = CustomUniverseProvider(["AAPL", "MSFT", "GOOGL"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from alphaevo.models.enums import MarketType
from alphaevo.models.market import StockInfo

if TYPE_CHECKING:
    from alphaevo.data.adapter import DataAdapter


class UniverseProvider(ABC):
    """Abstract interface for stock universe definition."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name."""

    @property
    def is_curated(self) -> bool:
        """Whether this is a manually curated (limited) universe.

        Returns True for demo/quickstart universes. Downstream code can
        use this flag to add disclaimers or adjust expectations.
        """
        return False

    @abstractmethod
    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Return the stock universe for a given market."""


class CuratedUniverseProvider(UniverseProvider):
    """Built-in curated universe for demos and quickstart.

    Uses a small, hand-picked set of liquid large-cap stocks that are
    reliable across data providers. Suitable for demos and development
    but NOT representative of full market research.
    """

    @property
    def name(self) -> str:
        return "curated"

    @property
    def is_curated(self) -> bool:
        return True

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Return curated stock list for the market."""
        # Import lazily to avoid circular dependency
        from alphaevo.data.adapters.yfinance import _STOCK_LISTS

        entries = _STOCK_LISTS.get(market, [])
        return [
            StockInfo(
                symbol=e["symbol"],
                name=e["name"],
                market=market,
                sector=e.get("sector"),
                market_cap=e.get("market_cap"),
                pe_ttm=e.get("pe_ttm"),
            )
            for e in entries
        ]


class AdapterUniverseProvider(UniverseProvider):
    """Universe provider backed by a DataAdapter.

    Delegates to the adapter's ``get_stock_list`` method, which may
    return dynamically fetched, full-market stock lists depending on
    the adapter implementation.
    """

    def __init__(self, adapter: DataAdapter) -> None:
        self._adapter = adapter

    @property
    def name(self) -> str:
        return f"adapter:{self._adapter.name}"

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        return await self._adapter.get_stock_list(market)


class CustomUniverseProvider(UniverseProvider):
    """User-supplied universe of specific symbols."""

    def __init__(self, symbols: list[str], market: MarketType = MarketType.US) -> None:
        self._symbols = symbols
        self._market = market

    @property
    def name(self) -> str:
        return f"custom({len(self._symbols)} symbols)"

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        if market != self._market:
            return []
        return [StockInfo(symbol=s, name=s, market=market) for s in self._symbols]
