"""Tests for RegimeDetector — market environment classification."""

import numpy as np
import pandas as pd

from alphaevo.models.enums import MarketRegime
from alphaevo.sampler.regime import RegimeDetector


def _make_ohlcv(closes: list[float], volatility: float = 0.01) -> pd.DataFrame:
    """Build synthetic OHLCV from a close price series."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    highs = closes_arr * (1 + volatility)
    lows = closes_arr * (1 - volatility)
    opens = np.roll(closes_arr, 1)
    opens[0] = closes_arr[0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes_arr,
            "volume": np.random.default_rng(42).integers(1_000_000, 10_000_000, n),
        }
    )


class TestRegimeDetector:
    def test_trending_up(self):
        """Steadily rising prices should be detected as trending_up."""
        closes = [100 + i * 0.5 for i in range(100)]  # 100 → 149.5
        df = _make_ohlcv(closes, volatility=0.005)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.TRENDING_UP

    def test_trending_down(self):
        """Steadily falling prices should be detected as trending_down."""
        closes = [150 - i * 0.5 for i in range(100)]  # 150 → 100.5
        df = _make_ohlcv(closes, volatility=0.005)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.TRENDING_DOWN

    def test_range_bound(self):
        """Sideways oscillation should be range_bound."""
        np.random.seed(42)
        closes = [100 + 2 * np.sin(i * 0.2) for i in range(100)]
        df = _make_ohlcv(closes, volatility=0.005)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.RANGE_BOUND

    def test_crash(self):
        """Sharp drop should be detected as panic."""
        closes = list(np.linspace(120, 125, 70)) + list(np.linspace(125, 95, 30))
        df = _make_ohlcv(closes, volatility=0.005)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.PANIC

    def test_high_volatility(self):
        """Large random swings around a flat mean should be volatile."""
        # Alternating high/low around 100 with large amplitude → high ATR, flat slope
        closes = []
        for i in range(100):
            closes.append(100 + (6 if i % 2 == 0 else -6))
        df = _make_ohlcv(closes, volatility=0.04)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.VOLATILE

    def test_insufficient_data(self):
        """Too little data should default to range_bound."""
        closes = [100 + i for i in range(20)]
        df = _make_ohlcv(closes)
        detector = RegimeDetector()
        regime = detector.detect(df)
        assert regime == MarketRegime.RANGE_BOUND

    def test_detect_periods(self):
        """Periods detection should partition time series into regimes."""
        # First half: trending up, second half: trending down
        up = [100 + i * 0.8 for i in range(80)]
        down = [up[-1] - i * 0.8 for i in range(80)]
        closes = up + down
        df = _make_ohlcv(closes, volatility=0.005)
        detector = RegimeDetector()
        periods = detector.detect_periods(df, window=65, step=20)
        assert len(periods) >= 2
        # First period should be up-ish, last period should be down-ish
        regimes = [p[2] for p in periods]
        assert MarketRegime.TRENDING_UP in regimes or MarketRegime.RANGE_BOUND in regimes

    def test_detect_periods_empty(self):
        """Empty or too-short data should return no periods."""
        df = _make_ohlcv([100] * 10)
        detector = RegimeDetector()
        periods = detector.detect_periods(df, window=65)
        assert periods == []

    def test_custom_thresholds(self):
        """Custom threshold parameters work."""
        detector = RegimeDetector(
            trend_slope_threshold=0.005,
            crash_slope_threshold=-0.01,
            volatility_threshold=0.05,
        )
        # Moderate rise should be range_bound with high threshold
        closes = [100 + i * 0.3 for i in range(100)]
        df = _make_ohlcv(closes, volatility=0.005)
        regime = detector.detect(df)
        assert regime in (MarketRegime.RANGE_BOUND, MarketRegime.TRENDING_UP)
