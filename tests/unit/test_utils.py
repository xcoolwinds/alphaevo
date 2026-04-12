"""Tests for alphaevo.utils — shared utility functions."""

from datetime import timezone

from alphaevo.utils import clamp, fmt_number, fmt_pct, safe_div, utcnow


class TestUtcnow:
    def test_returns_utc(self):
        now = utcnow()
        assert now.tzinfo is not None
        assert now.tzinfo == timezone.utc

    def test_type(self):
        from datetime import datetime

        assert isinstance(utcnow(), datetime)


class TestFmtPct:
    def test_basic(self):
        assert fmt_pct(0.423) == "42.3%"

    def test_zero(self):
        assert fmt_pct(0.0) == "0.0%"

    def test_negative(self):
        assert fmt_pct(-0.05) == "-5.0%"

    def test_custom_decimals(self):
        assert fmt_pct(0.12345, decimals=2) == "12.35%"

    def test_one_hundred_pct(self):
        assert fmt_pct(1.0) == "100.0%"


class TestFmtNumber:
    def test_basic(self):
        assert fmt_number(1234567.8) == "1,234,567.80"

    def test_zero(self):
        assert fmt_number(0) == "0.00"

    def test_custom_decimals(self):
        assert fmt_number(1000.5, decimals=0) == "1,000"


class TestClamp:
    def test_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_min(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_max(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0

    def test_at_boundary(self):
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


class TestSafeDiv:
    def test_normal(self):
        assert safe_div(10.0, 2.0) == 5.0

    def test_zero_denominator(self):
        assert safe_div(10.0, 0.0) == 0.0

    def test_custom_default(self):
        assert safe_div(10.0, 0.0, default=float("inf")) == float("inf")

    def test_negative(self):
        assert safe_div(-6.0, 3.0) == -2.0


class TestModuleExports:
    def test_all_exports(self):
        import alphaevo.utils

        assert hasattr(alphaevo.utils, "__all__")
        for name in alphaevo.utils.__all__:
            assert hasattr(alphaevo.utils, name)
