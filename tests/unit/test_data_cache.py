"""Tests for DataCache — Parquet-based data caching."""

import time
from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

from alphaevo.data.cache import DataCache


@pytest.fixture
def cache_dir(tmp_path):
    return tmp_path / "test_cache"


@pytest.fixture
def cache(cache_dir):
    return DataCache(cache_dir, ttl_seconds=2)


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000000] * 5,
        }
    )


class TestDataCache:
    def test_miss_on_empty_cache(self, cache):
        result = cache.get("AAPL", date(2024, 1, 1), date(2024, 3, 31))
        assert result is None

    def test_put_and_get(self, cache, sample_df):
        start, end = date(2024, 1, 1), date(2024, 1, 7)
        cache.put("AAPL", start, end, sample_df)
        result = cache.get("AAPL", start, end)
        assert result is not None
        assert len(result) == 5
        assert list(result.columns) == list(sample_df.columns)

    def test_different_symbols_isolated(self, cache, sample_df):
        start, end = date(2024, 1, 1), date(2024, 1, 7)
        cache.put("AAPL", start, end, sample_df)
        result = cache.get("MSFT", start, end)
        assert result is None

    def test_different_dates_isolated(self, cache, sample_df):
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        result = cache.get("AAPL", date(2024, 2, 1), date(2024, 2, 28))
        assert result is None

    def test_covering_range_can_serve_subset(self, cache, sample_df):
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        result = cache.get("AAPL", date(2024, 1, 5), date(2024, 1, 10))
        assert result is not None
        assert len(result) == len(sample_df)

    def test_prefers_smallest_covering_range(self, cache):
        wide_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=20, freq="B"),
                "open": [10.0] * 20,
                "high": [11.0] * 20,
                "low": [9.0] * 20,
                "close": [10.0] * 20,
                "volume": [1000] * 20,
            }
        )
        narrow_df = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-08", periods=5, freq="B"),
                "open": [20.0] * 5,
                "high": [21.0] * 5,
                "low": [19.0] * 5,
                "close": [20.0] * 5,
                "volume": [2000] * 5,
            }
        )
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), wide_df)
        cache.put("AAPL", date(2024, 1, 8), date(2024, 1, 12), narrow_df)

        result = cache.get("AAPL", date(2024, 1, 9), date(2024, 1, 11))

        assert result is not None
        assert set(result["close"]) == {20.0}

    def test_ttl_expiry_for_today(self, cache, sample_df):
        """Cache entries ending today should expire after TTL."""
        today = date.today()
        cache.put("AAPL", date(2024, 1, 1), today, sample_df)
        # Immediately: should hit
        assert cache.get("AAPL", date(2024, 1, 1), today) is not None
        # After TTL: should miss (we set ttl=2 seconds)
        time.sleep(2.1)
        assert cache.get("AAPL", date(2024, 1, 1), today) is None

    def test_no_ttl_for_historical(self, cache, sample_df):
        """Historical data (end date in past) never expires."""
        start, end = date(2023, 1, 1), date(2023, 12, 31)
        cache.put("AAPL", start, end, sample_df)
        time.sleep(2.1)
        assert cache.get("AAPL", start, end) is not None

    def test_put_empty_df_skipped(self, cache):
        start, end = date(2024, 1, 1), date(2024, 1, 7)
        cache.put("AAPL", start, end, pd.DataFrame())
        assert cache.get("AAPL", start, end) is None

    def test_put_none_skipped(self, cache):
        start, end = date(2024, 1, 1), date(2024, 1, 7)
        cache.put("AAPL", start, end, None)
        assert cache.get("AAPL", start, end) is None

    def test_invalidate_symbol(self, cache, sample_df):
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        cache.put("MSFT", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        removed = cache.invalidate("AAPL")
        assert removed == 1
        assert cache.get("AAPL", date(2024, 1, 1), date(2024, 1, 31)) is None
        assert cache.get("MSFT", date(2024, 1, 1), date(2024, 1, 31)) is not None

    def test_invalidate_all(self, cache, sample_df):
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        cache.put("MSFT", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        removed = cache.invalidate()
        assert removed == 2

    def test_invalidate_nonexistent(self, cache):
        removed = cache.invalidate("NOPE")
        assert removed == 0

    def test_stats(self, cache, sample_df):
        cache.put("AAPL", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        cache.get("AAPL", date(2024, 1, 1), date(2024, 1, 31))  # hit
        cache.get("MSFT", date(2024, 1, 1), date(2024, 1, 31))  # miss
        s = cache.stats()
        assert s["total_files"] == 1
        assert s["hits"] == 1
        assert s["misses"] == 1
        assert s["hit_rate"] == 0.5
        assert s["total_size_mb"] >= 0

    def test_special_chars_in_symbol(self, cache, sample_df):
        """Symbols with dots/slashes should be safely cached."""
        cache.put("600519.SS", date(2024, 1, 1), date(2024, 1, 31), sample_df)
        result = cache.get("600519.SS", date(2024, 1, 1), date(2024, 1, 31))
        assert result is not None

    def test_pickle_fallback_when_parquet_engine_unavailable(self, cache, sample_df):
        start, end = date(2024, 1, 1), date(2024, 1, 7)

        with patch.object(pd.DataFrame, "to_parquet", side_effect=ImportError("no engine")):
            cache.put("AAPL", start, end, sample_df)

        result = cache.get("AAPL", start, end)
        assert result is not None
        assert list(result["close"]) == list(sample_df["close"])
        stats = cache.stats()
        assert stats["total_files"] == 1

    def test_file_lock_prevents_corruption(self, cache, sample_df):
        """File lock context manager should acquire and release without error."""
        start, end = date(2024, 1, 1), date(2024, 1, 7)
        target = cache._cache_path("TEST", start, end, ".parquet")
        target.parent.mkdir(parents=True, exist_ok=True)

        # Lock should be acquirable and releasable
        with cache._file_lock(target):
            # Inside lock we can write directly (not via put, which also locks)
            sample_df.to_parquet(target, index=False)

        result = cache.get("TEST", start, end)
        assert result is not None
        assert len(result) == 5

    def test_lock_file_cleaned_up(self, cache, sample_df):
        """Lock files should be cleaned up after the context exits."""
        start, end = date(2024, 2, 1), date(2024, 2, 7)
        target = cache._cache_path("LOCK_TEST", start, end, ".parquet")
        lock_path = target.with_suffix(target.suffix + ".lock")
        target.parent.mkdir(parents=True, exist_ok=True)

        with cache._file_lock(target):
            pass

        # Lock file should be cleaned up (best-effort)
        assert not lock_path.exists()
