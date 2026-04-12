"""Tests for AdaptiveSampler."""

from __future__ import annotations

from datetime import date

import pytest

from alphaevo.models.enums import MarketRegime, MarketType, SamplingMethod, StrategyCategory
from alphaevo.models.market import StockInfo
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
    UniverseConfig,
    UniverseFilter,
)
from alphaevo.sampler.adaptive import AdaptiveSampler

# ── helpers ───────────────────────────────────────────────────────────


def _make_strategy(
    strategy_id: str = "test_strat_v1",
    filters: list[UniverseFilter] | None = None,
    category: StrategyCategory = StrategyCategory.TREND,
    preferred_regime: list[str] | None = None,
) -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=strategy_id,
            name="Test Strategy",
            market=MarketType.A_SHARE,
            category=category,
            preferred_regime=preferred_regime or [],
        ),
        description="A test strategy.",
        universe=UniverseConfig(
            market=["a_share_main"],
            filters=filters or [],
        ),
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
        ),
    )


def _make_stocks(
    n: int,
    sectors: list[str] | None = None,
    market_caps: list[float | None] | None = None,
) -> list[StockInfo]:
    """Create *n* mock StockInfo objects with optional sector/cap data."""
    stocks: list[StockInfo] = []
    for i in range(n):
        sector = sectors[i % len(sectors)] if sectors else None
        cap = market_caps[i % len(market_caps)] if market_caps else float((i + 1) * 1_000_000_000)
        stocks.append(
            StockInfo(
                symbol=f"{i:06d}.SZ",
                name=f"Stock {i}",
                market=MarketType.A_SHARE,
                sector=sector,
                market_cap=cap,
            )
        )
    return stocks


# ── tests ─────────────────────────────────────────────────────────────


class TestAdaptiveSampler:
    """Core sampler behaviour."""

    @pytest.mark.asyncio
    async def test_empty_stock_list(self) -> None:
        sampler = AdaptiveSampler(seed=42)
        batch = await sampler.sample(_make_strategy(), [])
        assert batch.symbols == []
        assert batch.strategy_id == "test_strat_v1"
        assert batch.sampling_reason == "empty stock list"

    @pytest.mark.asyncio
    async def test_representative_diverse_sectors(self) -> None:
        """Representative sampling should include stocks from each sector."""
        sectors = ["Tech", "Finance", "Health", "Energy"]
        stocks = _make_stocks(40, sectors=sectors)
        sampler = AdaptiveSampler(max_symbols=20, seed=42)
        batch = await sampler.sample(_make_strategy(), stocks, method=SamplingMethod.REPRESENTATIVE)

        selected_sectors = set()
        sym_to_sector = {s.symbol: s.sector for s in stocks}
        for sym in batch.symbols:
            selected_sectors.add(sym_to_sector[sym])

        assert len(batch.symbols) <= 20
        assert len(batch.symbols) >= 10
        # All sectors should be represented
        assert selected_sectors == set(sectors)

    @pytest.mark.asyncio
    async def test_max_symbols_respected(self) -> None:
        stocks = _make_stocks(100, sectors=["A", "B"])
        sampler = AdaptiveSampler(max_symbols=15, seed=0)
        batch = await sampler.sample(_make_strategy(), stocks)
        assert len(batch.symbols) <= 15

    @pytest.mark.asyncio
    async def test_min_symbols_respected(self) -> None:
        """When enough stocks exist, at least min_symbols are returned."""
        stocks = _make_stocks(50, sectors=["X"])
        sampler = AdaptiveSampler(min_symbols=10, max_symbols=30, seed=0)
        batch = await sampler.sample(_make_strategy(), stocks)
        assert len(batch.symbols) >= 10

    @pytest.mark.asyncio
    async def test_fewer_stocks_than_min(self) -> None:
        """If fewer stocks than min exist, return all of them."""
        stocks = _make_stocks(3)
        sampler = AdaptiveSampler(min_symbols=10, max_symbols=30, seed=0)
        batch = await sampler.sample(_make_strategy(), stocks)
        assert len(batch.symbols) == 3

    @pytest.mark.asyncio
    async def test_st_stocks_excluded(self) -> None:
        stocks = _make_stocks(20, sectors=["A"])
        stocks[0].is_st = True
        stocks[1].is_st = True
        sampler = AdaptiveSampler(seed=42)
        batch = await sampler.sample(_make_strategy(), stocks)
        assert stocks[0].symbol not in batch.symbols
        assert stocks[1].symbol not in batch.symbols


class TestStrategyScoped:
    """Strategy-scoped sampling applies universe filters first."""

    @pytest.mark.asyncio
    async def test_market_cap_filter(self) -> None:
        caps = [1e9, 5e9, 10e9, 20e9, 50e9]
        stocks = _make_stocks(25, sectors=["A", "B", "C", "D", "E"], market_caps=caps)
        strategy = _make_strategy(filters=[UniverseFilter(field="market_cap", op=">=", value=10e9)])
        sampler = AdaptiveSampler(max_symbols=30, min_symbols=5, seed=42)
        batch = await sampler.sample(strategy, stocks, method=SamplingMethod.STRATEGY_SCOPED)
        cap_lookup = {s.symbol: s.market_cap for s in stocks}
        for sym in batch.symbols:
            assert cap_lookup[sym] >= 10e9  # type: ignore[operator]

    @pytest.mark.asyncio
    async def test_filter_removes_all(self) -> None:
        stocks = _make_stocks(10, market_caps=[1e9])
        strategy = _make_strategy(
            filters=[UniverseFilter(field="market_cap", op=">=", value=100e9)]
        )
        sampler = AdaptiveSampler(seed=0)
        batch = await sampler.sample(strategy, stocks, method=SamplingMethod.STRATEGY_SCOPED)
        assert batch.symbols == []

    @pytest.mark.asyncio
    async def test_missing_field_skips_filter_when_no_metadata_available(self) -> None:
        stocks = _make_stocks(5, market_caps=[None])  # type: ignore[list-item]
        strategy = _make_strategy(filters=[UniverseFilter(field="market_cap", op=">=", value=1e9)])
        sampler = AdaptiveSampler(seed=0)
        batch = await sampler.sample(strategy, stocks, method=SamplingMethod.STRATEGY_SCOPED)
        assert sorted(batch.symbols) == sorted(stock.symbol for stock in stocks)


class TestSampleBatchMetadata:
    """Verify SampleBatch fields are populated correctly."""

    @pytest.mark.asyncio
    async def test_batch_id_contains_strategy_id(self) -> None:
        stocks = _make_stocks(15)
        sampler = AdaptiveSampler(seed=0)
        batch = await sampler.sample(_make_strategy(strategy_id="abc_v1"), stocks)
        assert batch.batch_id.startswith("abc_v1_")

    @pytest.mark.asyncio
    async def test_default_date_range(self) -> None:
        stocks = _make_stocks(15)
        sampler = AdaptiveSampler(seed=0)
        batch = await sampler.sample(_make_strategy(), stocks)
        today = date.today()
        assert batch.date_range[1] == today
        assert (today - batch.date_range[0]).days == 365

    @pytest.mark.asyncio
    async def test_custom_date_range(self) -> None:
        stocks = _make_stocks(15)
        sampler = AdaptiveSampler(seed=0)
        dr = (date(2024, 1, 1), date(2024, 12, 31))
        batch = await sampler.sample(_make_strategy(), stocks, date_range=dr)
        assert batch.date_range == dr

    @pytest.mark.asyncio
    async def test_sampling_method_recorded(self) -> None:
        stocks = _make_stocks(15)
        sampler = AdaptiveSampler(seed=0)
        batch = await sampler.sample(
            _make_strategy(), stocks, method=SamplingMethod.STRATEGY_SCOPED
        )
        assert batch.sampling_method == SamplingMethod.STRATEGY_SCOPED

    @pytest.mark.asyncio
    async def test_regime_based_prefers_large_caps_in_trending_up(self) -> None:
        caps = [float((i + 1) * 1_000_000_000) for i in range(18)]
        stocks = _make_stocks(18, sectors=["Tech"], market_caps=caps)
        sampler = AdaptiveSampler(max_symbols=6, min_symbols=3, seed=42)
        batch = await sampler.sample(
            _make_strategy(preferred_regime=["trending_up"]),
            stocks,
            method=SamplingMethod.REGIME_BASED,
            market_regime=MarketRegime.TRENDING_UP,
        )
        selected_caps = [stock.market_cap for stock in stocks if stock.symbol in batch.symbols]
        assert len(batch.symbols) == 6
        assert min(cap for cap in selected_caps if cap is not None) >= 10_000_000_000
        assert "trending_up" in batch.sampling_reason

    @pytest.mark.asyncio
    async def test_regime_based_prefers_smaller_caps_in_panic_for_event_strategy(self) -> None:
        caps = [float((i + 1) * 1_000_000_000) for i in range(18)]
        stocks = _make_stocks(18, sectors=["Event"], market_caps=caps)
        sampler = AdaptiveSampler(max_symbols=6, min_symbols=3, seed=42)
        batch = await sampler.sample(
            _make_strategy(
                category=StrategyCategory.EVENT,
                preferred_regime=["panic"],
            ),
            stocks,
            method=SamplingMethod.REGIME_BASED,
            market_regime=MarketRegime.PANIC,
        )
        selected_caps = [stock.market_cap for stock in stocks if stock.symbol in batch.symbols]
        assert len(batch.symbols) == 6
        assert max(cap for cap in selected_caps if cap is not None) <= 8_000_000_000

    @pytest.mark.asyncio
    async def test_strategy_scoped_uses_regime_bias_when_preferred_regime_present(self) -> None:
        caps = [1e9, 4e9, 8e9, 12e9, 16e9, 20e9]
        stocks = _make_stocks(24, sectors=["A", "B"], market_caps=caps)
        strategy = _make_strategy(
            filters=[UniverseFilter(field="market_cap", op=">=", value=4e9)],
            preferred_regime=["trending_up"],
        )
        sampler = AdaptiveSampler(max_symbols=6, min_symbols=3, seed=42)
        batch = await sampler.sample(
            strategy,
            stocks,
            method=SamplingMethod.STRATEGY_SCOPED,
            market_regime=MarketRegime.TRENDING_UP,
        )
        selected_caps = [stock.market_cap for stock in stocks if stock.symbol in batch.symbols]
        assert len(batch.symbols) == 6
        assert min(cap for cap in selected_caps if cap is not None) >= 8_000_000_000
        assert "regime-aware" in batch.sampling_reason
