"""Tests for DSAAdapter — daily_stock_analysis bridge (unit tests with mocks)."""

import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from alphaevo.models.enums import MarketRegime, MarketType


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
        mock_dsa_manager.get_daily_data.return_value = (mock_df, "MockFetcher")

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
        mock_dsa_manager.get_daily_data.assert_called_once_with(stock_code="000001", days=30)

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
        mock_dsa_manager.get_daily_data.return_value = mock_df

        df = await adapter.get_daily_data("000001", days=30)
        assert not df.empty
        assert "close" in df.columns
        assert len(df) == 3

    @pytest.mark.asyncio
    async def test_get_daily_data_empty(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_daily_data.return_value = (pd.DataFrame(), "MockFetcher")
        df = await adapter.get_daily_data("999999", days=30)
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_daily_data_error(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_daily_data.side_effect = Exception("connection error")
        df = await adapter.get_daily_data("000001", days=30)
        assert df.empty

    @pytest.mark.asyncio
    async def test_get_daily_data_legacy_fetch_daily(self, adapter, mock_dsa_manager):
        del mock_dsa_manager.get_daily_data
        mock_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-02", periods=2, freq="B"),
                "open": [10.0, 10.1],
                "high": [10.5, 10.6],
                "low": [9.5, 9.6],
                "close": [10.2, 10.3],
                "volume": [100000] * 2,
            }
        )
        mock_dsa_manager.fetch_daily.return_value = mock_df

        df = await adapter.get_daily_data("000001", days=30)
        assert len(df) == 2
        mock_dsa_manager.fetch_daily.assert_called_once_with("000001", 30)

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
    async def test_get_stock_list_dataframe_payload(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_stock_list.return_value = pd.DataFrame(
            {
                "代码": ["000001", "600519"],
                "名称": ["平安银行", "贵州茅台"],
                "总市值": ["210000000000", "2100000000000"],
                "市盈率": ["6.5", "28.0"],
            }
        )

        stocks = await adapter.get_stock_list(MarketType.A_SHARE)
        assert [s.symbol for s in stocks] == ["000001", "600519"]
        assert stocks[0].market_cap == 210000000000.0

    @pytest.mark.asyncio
    async def test_get_stock_list_us_returns_empty(self, adapter):
        stocks = await adapter.get_stock_list(MarketType.US)
        assert stocks == []

    @pytest.mark.asyncio
    async def test_get_stock_list_error(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_stock_list.side_effect = Exception("fail")
        stocks = await adapter.get_stock_list(MarketType.A_SHARE)
        assert stocks == []

    @pytest.mark.asyncio
    async def test_get_sector_data_uses_boards_and_rankings(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_belong_boards.return_value = [{"name": "白酒", "type": "行业"}]
        mock_dsa_manager.get_sector_rankings.return_value = (
            [{"name": "白酒", "change_pct": "3.2", "主力净流入": "120000000"}],
            [{"name": "煤炭", "change_pct": "-1.5"}],
        )

        sector = await adapter.get_sector_data("600519")

        assert sector is not None
        assert sector.name == "白酒"
        assert sector.heat_rank == 1
        assert sector.change_pct == pytest.approx(0.032)
        assert sector.net_inflow == 120000000.0
        assert not sector.risk_flag

    @pytest.mark.asyncio
    async def test_get_sector_data_marks_bottom_sector_risk(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_belong_boards.return_value = [{"name": "煤炭", "type": "行业"}]
        mock_dsa_manager.get_sector_rankings.return_value = (
            [{"name": "白酒", "change_pct": "3.2"}],
            [{"name": "煤炭", "change_pct": "-1.5"}],
        )

        sector = await adapter.get_sector_data("601088")

        assert sector is not None
        assert sector.name == "煤炭"
        assert sector.heat_rank == 2
        assert sector.risk_flag

    @pytest.mark.asyncio
    async def test_get_event_context_from_dsa_news_cache(self, adapter):
        from alphaevo.data.adapters import dsa as dsa_mod

        db = MagicMock()
        db.get_recent_news.return_value = [
            {
                "title": "公司预亏并收到监管问询",
                "snippet": "风险提示",
                "published_date": pd.Timestamp("2024-01-05"),
            },
            {
                "title": "公司宣布回购计划",
                "snippet": "利好",
                "published_date": pd.Timestamp("2024-01-08"),
            },
        ]
        storage = MagicMock()
        storage.get_db.return_value = db

        with patch.object(dsa_mod.importlib, "import_module", return_value=storage):
            context = await adapter.get_event_context(
                "600519",
                pd.Timestamp("2024-01-01").date(),
                pd.Timestamp("2024-01-31").date(),
            )

        assert context is not None
        assert context.source == "dsa_news_cache"
        assert len(context.records) == 2
        assert context.records[0].negative_news_score > context.records[1].negative_news_score

    @pytest.mark.asyncio
    async def test_get_market_context_from_dsa_market_payload(self, adapter, mock_dsa_manager):
        mock_dsa_manager.get_market_stats.return_value = {
            "up_count": 3600,
            "down_count": 900,
            "flat_count": 100,
            "limit_up_count": 80,
            "limit_down_count": 2,
        }
        mock_dsa_manager.get_main_indices.return_value = [
            {"code": "sh000001", "name": "上证指数", "change_pct": "1.2"}
        ]
        mock_dsa_manager.get_sector_rankings.return_value = (
            [{"name": "白酒"}],
            [{"name": "煤炭"}],
        )

        context = await adapter.get_market_context(MarketType.A_SHARE)

        assert context is not None
        assert context.index_change_pct == pytest.approx(0.012)
        assert context.breadth == pytest.approx(0.8)
        assert context.regime in {MarketRegime.TRENDING_UP, MarketRegime.EUPHORIA}
        assert context.sector_leaders == ["白酒"]
        assert context.sector_laggards == ["煤炭"]

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
    def test_load_dsa_supports_root_data_provider_package(self, tmp_path, monkeypatch):
        pkg = tmp_path / "data_provider"
        pkg.mkdir()
        (pkg / "__init__.py").write_text(
            "class DataFetcherManager:\n    pass\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(sys, "path", [p for p in sys.path if str(tmp_path) not in p])
        sys.modules.pop("data_provider", None)
        sys.modules.pop("data_provider.base", None)

        from alphaevo.data.adapters.dsa import _load_dsa

        manager_cls = _load_dsa(str(tmp_path))

        assert manager_cls.__name__ == "DataFetcherManager"
        sys.modules.pop("data_provider", None)
        sys.modules.pop("data_provider.base", None)

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
