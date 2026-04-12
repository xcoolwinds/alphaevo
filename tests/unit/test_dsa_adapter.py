"""Tests for DSAAdapter — daily_stock_analysis bridge (unit tests with mocks)."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphaevo.models.enums import MarketType


class TestDSAAdapter:
    """Test DSAAdapter with mocked DSA internals."""

    @pytest.fixture
    def mock_dsa_manager(self):
        """Return a mock DataFetcherManager."""
        return MagicMock()

    @pytest.fixture
    def adapter(self, mock_dsa_manager):
        """Create a DSAAdapter with mocked _load_dsa."""
        with patch("alphaevo.data.adapters.dsa._load_dsa") as mock_load:
            mock_load.return_value = lambda: mock_dsa_manager
            from alphaevo.data.adapters.dsa import DSAAdapter

            inst = DSAAdapter.__new__(DSAAdapter)
            inst._manager = mock_dsa_manager
            return inst

    def test_name(self, adapter):
        assert adapter.name == "dsa"

    @pytest.mark.asyncio
    async def test_get_daily_data_success(self, adapter, mock_dsa_manager):
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=5, freq="B"),
                "open": [10.0, 10.1, 10.2, 10.3, 10.4],
                "high": [10.5, 10.6, 10.7, 10.8, 10.9],
                "low": [9.5, 9.6, 9.7, 9.8, 9.9],
                "close": [10.2, 10.3, 10.4, 10.5, 10.6],
                "volume": [100000] * 5,
            }
        )
        mock_dsa_manager.fetch_daily.return_value = mock_df

        df = await adapter.get_daily_data("000001", days=30)
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

    @pytest.mark.asyncio
    async def test_get_daily_data_chinese_cols(self, adapter, mock_dsa_manager):
        """DSA may return Chinese column names."""
        mock_df = pd.DataFrame(
            {
                "日期": pd.date_range("2024-01-02", periods=3, freq="B"),
                "开盘": [10.0, 10.1, 10.2],
                "最高": [10.5, 10.6, 10.7],
                "最低": [9.5, 9.6, 9.7],
                "收盘": [10.2, 10.3, 10.4],
                "成交量": [100000] * 3,
            }
        )
        mock_dsa_manager.fetch_daily.return_value = mock_df

        df = await adapter.get_daily_data("000001", days=30)
        assert not df.empty
        assert "close" in df.columns
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_get_daily_data_empty(self, adapter, mock_dsa_manager):
        mock_dsa_manager.fetch_daily.return_value = pd.DataFrame()
        df = await adapter.get_daily_data("999999", days=30)
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_daily_data_error(self, adapter, mock_dsa_manager):
        mock_dsa_manager.fetch_daily.side_effect = Exception("connection error")
        df = await adapter.get_daily_data("000001", days=30)
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_stock_list_a_share(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_stock_list.return_value = [
            {"code": "000001", "name": "平安银行"},
            {"code": "600519", "name": "贵州茅台"},
        ]
        stocks = await adapter.get_stock_list(MarketType.A_SHARE)
        assert len(stocks) == 2
        assert stocks[0].symbol == "000001"
        assert stocks[0].market == MarketType.A_SHARE

    @pytest.mark.asyncio
    async def test_get_stock_list_us_returns_empty(self, adapter):
        stocks = await adapter.get_stock_list(MarketType.US)
        assert stocks == []

    @pytest.mark.asyncio
    async def test_get_stock_list_error(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_stock_list.side_effect = Exception("fail")
        stocks = await adapter.get_stock_list(MarketType.A_SHARE)
        assert stocks == []

    def test_normalize_empty_df(self):
        from alphaevo.data.adapters.dsa import DSAAdapter

        df = DSAAdapter._normalize(pd.DataFrame({"date": [], "open": [], "close": []}))
        assert df.empty

    def test_empty_df_schema(self):
        from alphaevo.data.adapters.dsa import DSAAdapter

        df = DSAAdapter._empty_df()
        assert list(df.columns) == [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "prev_close",
        ]


class TestDSALoadErrors:
    def test_missing_env_var(self):
        with patch.dict("os.environ", {}, clear=True):
            # Remove ALPHAEVO_DSA_PATH if present
            import os

            os.environ.pop("ALPHAEVO_DSA_PATH", None)

            from alphaevo.data.adapters.dsa import _load_dsa

            with pytest.raises(ImportError, match="ALPHAEVO_DSA_PATH"):
                _load_dsa(None)

    def test_nonexistent_path(self, tmp_path):
        fake_path = str(tmp_path / "does_not_exist")
        from alphaevo.data.adapters.dsa import _load_dsa

        with pytest.raises(ImportError, match="does not exist"):
            _load_dsa(fake_path)
