"""Market data models."""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field

from alphaevo.models.enums import MarketRegime, MarketType


class PriceData(BaseModel):
    """OHLC price data."""

    open: float
    high: float
    low: float
    close: float
    prev_close: float | None = None
    change_pct: float | None = None


class VolumeData(BaseModel):
    """Volume and turnover data."""

    volume: float  # 成交量（股）
    amount: float | None = None  # 成交额（元）
    turnover_rate: float | None = None  # 换手率


class TechnicalIndicators(BaseModel):
    """Common technical indicators."""

    ma5: float | None = None
    ma10: float | None = None
    ma20: float | None = None
    ma60: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    rsi_6: float | None = None
    rsi_14: float | None = None
    boll_upper: float | None = None
    boll_middle: float | None = None
    boll_lower: float | None = None
    kdj_k: float | None = None
    kdj_d: float | None = None
    kdj_j: float | None = None
    atr: float | None = None
    volume_ratio_1d_5d: float | None = None
    relative_strength_20d: float | None = None


class FundamentalData(BaseModel):
    """Fundamental financial data."""

    pe_ttm: float | None = None
    pb: float | None = None
    ps_ttm: float | None = None
    roe: float | None = None
    market_cap: float | None = None  # 总市值
    free_float_cap: float | None = None  # 流通市值
    revenue_yoy: float | None = None
    profit_yoy: float | None = None


class NewsItem(BaseModel):
    """A single news item."""

    title: str
    source: str | None = None
    published_at: datetime | None = None
    sentiment_score: float | None = None  # -1.0 ~ 1.0
    url: str | None = None
    summary: str | None = None


class EventContextRecord(BaseModel):
    """Provider-supplied event/news context aligned to a specific trading date."""

    date: date
    negative_news_score: float | None = None
    news_sentiment_score: float | None = None
    days_since_event: int | None = None
    pre_event_close: float | None = None
    price_above_pre_event: bool | None = None
    already_overreacted: bool | None = None


class EventContextSeries(BaseModel):
    """A date-aligned event/news context series for one symbol."""

    symbol: str
    source: str = "provider"
    records: list[EventContextRecord] = Field(default_factory=list)


class MarketContext(BaseModel):
    """Overall market environment snapshot."""

    index_change_pct: float | None = None  # 大盘涨跌幅
    breadth: float | None = None  # 涨跌家数比
    regime: MarketRegime | None = None
    sentiment_index: float | None = None  # 市场情绪指数 0~1
    sector_leaders: list[str] = Field(default_factory=list)
    sector_laggards: list[str] = Field(default_factory=list)


class MarketSnapshot(BaseModel):
    """Complete market snapshot for a single stock on a given date."""

    symbol: str
    name: str = ""
    date: date
    market: MarketType

    price: PriceData
    volume: VolumeData
    indicators: TechnicalIndicators = Field(default_factory=TechnicalIndicators)
    fundamentals: FundamentalData | None = None
    news: list[NewsItem] = Field(default_factory=list)
    sector: str | None = None
    market_context: MarketContext | None = None


class RealTimeQuote(BaseModel):
    """Real-time stock quote."""

    symbol: str
    name: str = ""
    price: float
    change_pct: float
    volume: float
    amount: float | None = None
    timestamp: datetime | None = None


class StockInfo(BaseModel):
    """Basic stock information."""

    symbol: str
    name: str
    market: MarketType
    sector: str | None = None
    market_cap: float | None = None
    pe_ttm: float | None = None
    is_st: bool = False


class SectorInfo(BaseModel):
    """Sector/industry information."""

    name: str
    change_pct: float
    heat_rank: int | None = None
    rising_days: int | None = None
    net_inflow: float | None = None
    net_inflow_days: int | None = None
    risk_flag: bool = False
    top_stocks: list[str] = Field(default_factory=list)


class IndicatorContext(BaseModel):
    """Context passed to indicator functions alongside the OHLCV DataFrame.

    Carries auxiliary data that pure OHLCV can't provide — benchmark index,
    sector info, stock metadata, etc. Indicators that only need OHLCV can
    ignore this object entirely.
    """

    model_config = {"arbitrary_types_allowed": True}

    benchmark_df: object | None = None  # pd.DataFrame of benchmark index
    sector_info: SectorInfo | None = None
    stock_info: StockInfo | None = None
    market_context: MarketContext | None = None
    intra_sector_strength_rank_pct: float | None = None
    negative_news_score: float | None = None
    news_sentiment_score: float | None = None
    days_since_event: int | None = None
    pre_event_close: float | None = None
    price_above_pre_event: bool | None = None
    already_overreacted: bool | None = None
    event_context_source: str | None = None
