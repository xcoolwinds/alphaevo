"""Tests for AkShareAdapter — A-share data adapter (unit tests with mocks)."""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from alphaevo.data.adapters.akshare import AkShareAdapter, _normalize_a_share_symbol
from alphaevo.models.enums import MarketType


class TestSymbolNormalization:
    def test_plain_code(self):
        assert _normalize_a_share_symbol("000001") == "000001"

    def test_sz_suffix(self):
        assert _normalize_a_share_symbol("000001.SZ") == "000001"

    def test_ss_suffix(self):
        assert _normalize_a_share_symbol("600519.SS") == "600519"

    def test_sh_suffix(self):
        assert _normalize_a_share_symbol("600519.SH") == "600519"

    def test_lowercase_prefix(self):
        assert _normalize_a_share_symbol("sz000001") == "000001"

    def test_uppercase_prefix(self):
        assert _normalize_a_share_symbol("SH600519") == "600519"

    def test_whitespace(self):
        assert _normalize_a_share_symbol("  000001  ") == "000001"


class TestAkShareAdapter:
    @pytest.fixture
    def adapter(self):
        return AkShareAdapter()

    def test_name(self, adapter):
        assert adapter.name == "akshare"

    @pytest.mark.asyncio
    async def test_get_daily_data_success(self, adapter):
        """Mock akshare to return valid data."""
        mock_df = pd.DataFrame(
            {
                "日期": pd.date_range("2024-01-02", periods=5, freq="B"),
                "开盘": [10.0, 10.1, 10.2, 10.3, 10.4],
                "最高": [10.5, 10.6, 10.7, 10.8, 10.9],
                "最低": [9.5, 9.6, 9.7, 9.8, 9.9],
                "收盘": [10.2, 10.3, 10.4, 10.5, 10.6],
                "成交量": [100000] * 5,
            }
        )

        with patch.object(adapter, "_fetch_hist", return_value=mock_df):
            df = await adapter.get_daily_data("000001.SZ", days=30)
            assert not df.empty
            assert list(df.columns) == [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "prev_close",
            ]
            assert len(df) == 5
            assert df["close"].iloc[-1] == 10.6

    @pytest.mark.asyncio
    async def test_get_daily_data_empty(self, adapter):
        """Empty response returns empty df."""
        with patch.object(adapter, "_fetch_hist", return_value=pd.DataFrame()):
            df = await adapter.get_daily_data("999999", days=30)
            assert df.empty

    @pytest.mark.asyncio
    async def test_get_daily_data_error(self, adapter):
        """Exception returns empty df."""
        with patch.object(adapter, "_fetch_hist", side_effect=Exception("net error")):
            df = await adapter.get_daily_data("000001", days=30)
            assert df.empty

    @pytest.mark.asyncio
    async def test_get_stock_list_a_share(self, adapter):
        """Mock akshare spot data for stock list."""
        pd.DataFrame(
            {
                "代码": ["000001", "600519", "000858"],
                "名称": ["平安银行", "贵州茅台", "五粮液"],
            }
        )
        with patch.object(adapter, "_fetch_stock_list") as mock_fetch:
            from alphaevo.models.market import StockInfo

            mock_fetch.return_value = [
                StockInfo(symbol="000001", name="平安银行", market=MarketType.A_SHARE),
                StockInfo(symbol="600519", name="贵州茅台", market=MarketType.A_SHARE),
            ]
            stocks = await adapter.get_stock_list(MarketType.A_SHARE)
            assert len(stocks) == 2
            assert stocks[0].symbol == "000001"

    @pytest.mark.asyncio
    async def test_get_stock_list_us_returns_empty(self, adapter):
        """Non-A-share markets return empty."""
        stocks = await adapter.get_stock_list(MarketType.US)
        assert stocks == []

    @pytest.mark.asyncio
    async def test_get_index_data(self, adapter):
        """Mock index data fetch."""
        mock_df = pd.DataFrame(
            {
                "日期": pd.date_range("2024-01-02", periods=3, freq="B"),
                "开盘": [3000.0, 3010.0, 3020.0],
                "最高": [3050.0, 3060.0, 3070.0],
                "最低": [2980.0, 2990.0, 3000.0],
                "收盘": [3020.0, 3030.0, 3050.0],
                "成交量": [500000000] * 3,
            }
        )
        with patch.object(adapter, "_fetch_index", return_value=mock_df):
            df = await adapter.get_index_data("000300", date(2024, 1, 1), date(2024, 1, 5))
            assert not df.empty
            assert "close" in df.columns
