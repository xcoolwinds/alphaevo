"""DSA (daily_stock_analysis) optional bridge for AlphaEvo.

Bridges AlphaEvo to the ``daily_stock_analysis`` project's
``DataFetcherManager``, reusing its multi-source fallback capability.

This module is intentionally treated as an external enhancement bridge rather
than a core built-in dependency of AlphaEvo.

Install requirements:

1. Clone or install daily_stock_analysis separately.
2. Set the environment variable ``ALPHAEVO_DSA_PATH`` to the project root.
3. Use the ``dsa`` adapter::

       ALPHAEVO_DATA_ADAPTER=dsa alphaevo run <strategy>

This adapter isolates AlphaEvo from DSA internals through the
:class:`DataAdapter` interface — no direct imports leak outside this file.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from alphaevo.data.adapter import DataAdapter
from alphaevo.models.enums import MarketType
from alphaevo.models.market import StockInfo

logger = logging.getLogger(__name__)


def _to_float(value: Any) -> float | None:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _load_dsa(dsa_path: str | None = None) -> Any:
    """Import the DSA DataFetcherManager, injecting *dsa_path* into sys.path if needed."""
    import os

    path = dsa_path or os.getenv("ALPHAEVO_DSA_PATH", "")
    if not path:
        raise ImportError(
            "ALPHAEVO_DSA_PATH environment variable is required for DSAAdapter. "
            "Point it to the root of your daily_stock_analysis clone."
        )

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir():
        raise ImportError(f"DSA path does not exist: {resolved}")

    src_dir = str(resolved / "src") if (resolved / "src").is_dir() else str(resolved)
    added_to_path = False
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        added_to_path = True

    try:
        mod = importlib.import_module("data_fetcher.manager")
        return mod.DataFetcherManager
    except (ImportError, ModuleNotFoundError) as err:
        # Clean up sys.path on failure
        if added_to_path and src_dir in sys.path:
            sys.path.remove(src_dir)
        raise ImportError(
            f"Cannot import DataFetcherManager from {resolved}. "
            "Ensure daily_stock_analysis is installed correctly."
        ) from err


class DSAAdapter(DataAdapter):
    """Data adapter that delegates to daily_stock_analysis's DataFetcherManager."""

    def __init__(self, dsa_path: str | None = None) -> None:
        manager_cls = _load_dsa(dsa_path)
        self._manager = manager_cls()

    @property
    def name(self) -> str:  # noqa: D401
        return "dsa"

    async def get_daily_data(self, symbol: str, days: int = 120) -> pd.DataFrame:
        """Fetch daily OHLCV via DSA's multi-source fetcher."""
        try:
            df = await asyncio.to_thread(self._manager.fetch_daily, symbol, days)
        except Exception:
            logger.exception("DSA fetch failed for %s", symbol)
            return self._empty_df()

        if df is None or df.empty:
            logger.warning("No data returned from DSA for %s", symbol)
            return self._empty_df()

        return self._normalize(df)

    async def get_stock_list(self, market: MarketType) -> list[StockInfo]:
        """Delegate stock list retrieval to DSA."""
        if market != MarketType.A_SHARE:
            return []

        try:
            raw = await asyncio.to_thread(self._manager.get_stock_list)
            if not raw:
                return []

            stocks: list[StockInfo] = []
            for item in raw[:500]:
                code = str(item.get("code", item.get("symbol", ""))).strip()
                name = str(item.get("name", "")).strip()
                if code:
                    stocks.append(
                        StockInfo(
                            symbol=code,
                            name=name,
                            market=MarketType.A_SHARE,
                            sector=item.get("sector"),
                            market_cap=_to_float(item.get("market_cap")),
                            pe_ttm=_to_float(item.get("pe_ttm")),
                        )
                    )
            return stocks
        except Exception:
            logger.exception("DSA stock list fetch failed")
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(df: pd.DataFrame) -> pd.DataFrame:
        """Map DSA output to AlphaEvo's standard OHLCV schema."""
        df = df.copy()

        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }

        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        for col in ["date", "open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                if col == "volume":
                    df[col] = 0
                else:
                    return DSAAdapter._empty_df()

        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date").reset_index(drop=True)
        df["prev_close"] = df["close"].shift(1)

        return df[["date", "open", "high", "low", "close", "volume", "prev_close"]].copy()

    @staticmethod
    def _empty_df() -> pd.DataFrame:
        """Return an empty DataFrame with the standard column schema."""
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "volume", "prev_close"]
        )
