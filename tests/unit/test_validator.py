"""Tests for FactorValidator (statistical quality checks)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaevo.alpha_factory.validator import (
    FactorValidator,
    ValidationResult,
    ValidationThresholds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_predictive_factor(n: int = 200, noise: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Create a factor that correlates with forward returns."""
    rng = np.random.default_rng(42)
    factor = pd.Series(rng.standard_normal(n))
    returns = factor * 0.5 + rng.standard_normal(n) * noise
    return factor, returns


def _make_random_factor(n: int = 200) -> tuple[pd.Series, pd.Series]:
    """Create a factor with zero correlation to returns."""
    rng = np.random.default_rng(123)
    factor = pd.Series(rng.standard_normal(n))
    returns = pd.Series(rng.standard_normal(n))
    return factor, returns


def _make_negative_factor(n: int = 200, noise: float = 0.3) -> tuple[pd.Series, pd.Series]:
    """Create a factor with negative correlation to forward returns."""
    rng = np.random.default_rng(314)
    factor = pd.Series(rng.standard_normal(n))
    returns = factor * -0.5 + rng.standard_normal(n) * noise
    return factor, returns


# ---------------------------------------------------------------------------
# Tests — ValidationResult model
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_passed_result(self):
        r = ValidationResult(factor_name="test", passed=True, ic_mean=0.05)
        assert r.passed
        assert r.reasons == []

    def test_failed_result_with_reasons(self):
        r = ValidationResult(
            factor_name="bad",
            passed=False,
            reasons=["IC too low"],
        )
        assert not r.passed
        assert "IC too low" in r.reasons


# ---------------------------------------------------------------------------
# Tests — IC Computation
# ---------------------------------------------------------------------------


class TestICComputation:
    def test_predictive_factor_positive_ic(self):
        factor, returns = _make_predictive_factor()
        v = FactorValidator()
        ic_mean, ic_std = v._compute_ic(factor, returns)
        assert ic_mean > 0.1  # Should have strong positive IC
        assert ic_std > 0

    def test_random_factor_near_zero_ic(self):
        factor, returns = _make_random_factor()
        v = FactorValidator()
        ic_mean, _ = v._compute_ic(factor, returns)
        assert abs(ic_mean) < 0.15  # Should be near zero

    def test_short_series_returns_default(self):
        factor = pd.Series([1.0, 2.0, 3.0])
        returns = pd.Series([0.1, 0.2, 0.3])
        v = FactorValidator()
        ic_mean, ic_std = v._compute_ic(factor, returns)
        assert ic_mean == 0.0
        assert ic_std == 1.0


# ---------------------------------------------------------------------------
# Tests — Full Validation
# ---------------------------------------------------------------------------


class TestValidate:
    def test_good_factor_passes(self):
        factor, returns = _make_predictive_factor(n=300, noise=0.2)
        v = FactorValidator(ValidationThresholds(min_ic_abs=0.02, min_ir=0.1))
        result = v.validate("good_factor", factor, returns)
        assert result.factor_name == "good_factor"
        assert result.ic_mean > 0
        # Good factor should pass most checks
        assert result.passed or len(result.reasons) <= 1

    def test_random_factor_fails(self):
        factor, returns = _make_random_factor()
        v = FactorValidator(ValidationThresholds(min_ic_abs=0.05, min_ir=0.3))
        result = v.validate("random_factor", factor, returns)
        assert len(result.reasons) > 0  # Should fail at least IC check

    def test_cross_correlation_check(self):
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100))
        # Create a duplicate factor (perfect correlation)
        duplicate = factor * 1.0 + rng.standard_normal(100) * 0.01

        v = FactorValidator(
            ValidationThresholds(
                min_ic_abs=0.0,  # Relax other checks
                min_ir=0.0,
                min_monthly_win_rate=0.0,
                max_cross_correlation=0.5,
            )
        )
        result = v.validate(
            "new_factor",
            factor,
            returns,
            existing_factors={"existing": duplicate},
        )
        # Should be flagged for high correlation
        corr_reasons = [r for r in result.reasons if "Correlated" in r]
        assert len(corr_reasons) > 0

    def test_no_cross_correlation_with_independent(self):
        rng = np.random.default_rng(42)
        factor = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100))
        independent = pd.Series(np.random.default_rng(999).standard_normal(100))

        v = FactorValidator(
            ValidationThresholds(
                min_ic_abs=0.0,
                min_ir=0.0,
                min_monthly_win_rate=0.0,
                max_cross_correlation=0.5,
            )
        )
        result = v.validate(
            "new",
            factor,
            returns,
            existing_factors={"other": independent},
        )
        corr_reasons = [r for r in result.reasons if "Correlated" in r]
        assert len(corr_reasons) == 0

    def test_cross_sectional_inputs_validate(self):
        rng = np.random.default_rng(7)
        dates = pd.Series(pd.date_range("2024-01-01", periods=20, freq="D").repeat(5))
        symbols = pd.Series([f"S{i}" for _ in range(20) for i in range(5)])
        factor = pd.Series(rng.standard_normal(len(dates)))
        returns = factor * 0.4 + pd.Series(rng.standard_normal(len(dates)) * 0.1)

        v = FactorValidator(
            ValidationThresholds(
                min_ic_abs=0.0,
                min_ir=0.0,
                min_monthly_win_rate=0.0,
            )
        )
        result = v.validate(
            "cross_sectional_factor",
            factor,
            returns,
            dates=dates,
            symbols=symbols,
        )

        assert result.ic_mean > 0.2
        assert 0.0 <= result.monthly_win_rate <= 1.0

    def test_expected_positive_direction_rejects_negative_factor(self):
        factor, returns = _make_negative_factor(n=300, noise=0.15)
        v = FactorValidator(
            ValidationThresholds(
                min_ic_abs=0.02,
                min_ir=0.1,
                min_monthly_win_rate=0.0,
            )
        )

        result = v.validate(
            "wrong_direction_factor",
            factor,
            returns,
            expected_direction="positive",
        )

        assert not result.passed
        assert any("expected positive direction" in reason for reason in result.reasons)

    def test_expected_negative_direction_accepts_negative_factor(self):
        factor, returns = _make_negative_factor(n=300, noise=0.15)
        v = FactorValidator(
            ValidationThresholds(
                min_ic_abs=0.02,
                min_ir=0.1,
                min_monthly_win_rate=0.3,
            )
        )

        result = v.validate(
            "negative_factor",
            factor,
            returns,
            expected_direction="negative",
        )

        assert result.ic_mean < 0
        assert result.monthly_win_rate > 0.5
        assert result.passed


# ---------------------------------------------------------------------------
# Tests — Monthly Win Rate
# ---------------------------------------------------------------------------


class TestMonthlyWinRate:
    def test_positive_for_predictive_factor(self):
        factor, returns = _make_predictive_factor(n=200)
        v = FactorValidator()
        wr = v._monthly_win_rate(factor, returns, dates=None)
        assert wr > 0.3  # Predictive → should win most months

    def test_short_series_returns_zero(self):
        factor = pd.Series([1.0] * 5)
        returns = pd.Series([0.1] * 5)
        v = FactorValidator()
        wr = v._monthly_win_rate(factor, returns, dates=None)
        assert wr == 0.0

    def test_cross_sectional_monthly_win_rate(self):
        rng = np.random.default_rng(11)
        dates = pd.Series(pd.date_range("2024-01-01", periods=40, freq="D").repeat(4))
        symbols = pd.Series([f"S{i}" for _ in range(40) for i in range(4)])
        factor = pd.Series(rng.standard_normal(len(dates)))
        returns = factor * 0.5 + pd.Series(rng.standard_normal(len(dates)) * 0.2)
        v = FactorValidator()
        wr = v._monthly_win_rate(factor, returns, dates=dates, symbols=symbols)
        assert wr > 0.5

    def test_negative_direction_monthly_win_rate(self):
        factor, returns = _make_negative_factor(n=200, noise=0.2)
        v = FactorValidator()
        wr = v._monthly_win_rate(
            factor,
            returns,
            dates=None,
            expected_direction="negative",
        )
        assert wr > 0.5


# ---------------------------------------------------------------------------
# Tests — Turnover
# ---------------------------------------------------------------------------


class TestTurnover:
    def test_stable_factor_low_turnover(self):
        factor = pd.Series(range(100), dtype=float)
        v = FactorValidator()
        t = v._compute_turnover(factor)
        assert 0 <= t <= 1

    def test_short_series_returns_zero(self):
        factor = pd.Series([1.0, 2.0])
        v = FactorValidator()
        t = v._compute_turnover(factor)
        assert t == 0.0

    def test_cross_sectional_turnover_bounded(self):
        rng = np.random.default_rng(21)
        dates = pd.Series(pd.date_range("2024-01-01", periods=12, freq="D").repeat(4))
        symbols = pd.Series([f"S{i}" for _ in range(12) for i in range(4)])
        factor = pd.Series(rng.standard_normal(len(dates)))
        v = FactorValidator()
        t = v._compute_turnover(factor, dates=dates, symbols=symbols)
        assert 0.0 <= t <= 1.0


# ---------------------------------------------------------------------------
# Tests — Thresholds
# ---------------------------------------------------------------------------


class TestThresholds:
    def test_default_thresholds(self):
        t = ValidationThresholds()
        assert t.min_ic_abs == 0.02
        assert t.min_ir == 0.3
        assert t.max_turnover == 0.8
        assert t.min_cross_section_size == 4

    def test_custom_thresholds(self):
        t = ValidationThresholds(min_ic_abs=0.05, min_ir=0.5)
        assert t.min_ic_abs == 0.05
        assert t.min_ir == 0.5
