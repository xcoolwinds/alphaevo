"""Tests for Alpha Factory: dynamic registry + factory orchestrator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore
from alphaevo.alpha_factory.factory import (
    AlphaFactory,
    DiscoveryResult,
    load_registered_factors,
)
from alphaevo.alpha_factory.synthesizer import FactorHypothesis
from alphaevo.backtest.indicators import IndicatorRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame(
        {
            "open": close + rng.uniform(-0.5, 0.5, n),
            "high": close + abs(rng.standard_normal(n)),
            "low": close - abs(rng.standard_normal(n)),
            "close": close,
            "volume": rng.integers(1000, 10000, n).astype(float),
        }
    )


_GOOD_CODE = "def compute(df, idx):\n    return float(df['close'].iloc[idx])"

_VALID_FACTOR = {
    "name": "test_dynamic_factor",
    "description": "Test factor",
    "rationale": "Testing",
    "code": _GOOD_CODE,
    "expected_direction": "positive",
}


class _MockLLM:
    def __init__(self, response=None, raise_error=False):
        self._response = response
        self._raise = raise_error

    def chat_json(self, messages, **kwargs):
        if self._raise:
            raise RuntimeError("LLM error")
        return self._response


# ---------------------------------------------------------------------------
# Tests — Dynamic Registry
# ---------------------------------------------------------------------------


class TestDynamicRegistry:
    def setup_method(self):
        # Clean up any dynamic registrations from previous tests
        for name in list(IndicatorRegistry._dynamic_registry.keys()):
            IndicatorRegistry.unregister_dynamic(name)

    def teardown_method(self):
        for name in list(IndicatorRegistry._dynamic_registry.keys()):
            IndicatorRegistry.unregister_dynamic(name)

    def test_register_dynamic(self):
        def my_factor(df, idx):
            return 42.0

        IndicatorRegistry.register_dynamic("test_dyn_1", my_factor)
        assert IndicatorRegistry.is_registered("test_dyn_1")
        assert "test_dyn_1" in IndicatorRegistry.dynamic_names()

        df = _make_df(5)
        val = IndicatorRegistry.compute("test_dyn_1", df, 0)
        assert val == 42.0

    def test_unregister_dynamic(self):
        def my_factor(df, idx):
            return 1.0

        IndicatorRegistry.register_dynamic("test_dyn_2", my_factor)
        assert IndicatorRegistry.is_registered("test_dyn_2")

        removed = IndicatorRegistry.unregister_dynamic("test_dyn_2")
        assert removed is True
        assert not IndicatorRegistry.is_registered("test_dyn_2")
        assert "test_dyn_2" not in IndicatorRegistry.dynamic_names()

    def test_unregister_nonexistent(self):
        removed = IndicatorRegistry.unregister_dynamic("no_such")
        assert removed is False

    def test_dynamic_names_empty(self):
        assert IndicatorRegistry.dynamic_names() == []

    def test_dynamic_in_available(self):
        def my_factor(df, idx):
            return 0.0

        IndicatorRegistry.register_dynamic("test_dyn_avail", my_factor)
        assert "test_dyn_avail" in IndicatorRegistry.available()


# ---------------------------------------------------------------------------
# Tests — DiscoveryResult
# ---------------------------------------------------------------------------


class TestDiscoveryResult:
    def test_empty_result(self):
        r = DiscoveryResult()
        assert r.success_count == 0
        assert "Proposed: 0" in r.summary()

    def test_with_data(self):
        r = DiscoveryResult()
        r.proposed = [FactorHypothesis(**_VALID_FACTOR)]
        r.sandbox_passed = ["f1"]
        r.registered = ["f1"]
        assert r.success_count == 1
        assert "Registered: 1" in r.summary()


# ---------------------------------------------------------------------------
# Tests — AlphaFactory pipeline
# ---------------------------------------------------------------------------


class TestAlphaFactory:
    def setup_method(self):
        for name in list(IndicatorRegistry._dynamic_registry.keys()):
            IndicatorRegistry.unregister_dynamic(name)

    def teardown_method(self):
        for name in list(IndicatorRegistry._dynamic_registry.keys()):
            IndicatorRegistry.unregister_dynamic(name)

    @pytest.mark.asyncio
    async def test_discover_no_hypotheses(self):
        llm = _MockLLM(response=[])
        factory = AlphaFactory(llm)
        df = _make_df(50)
        returns = pd.Series(np.random.default_rng(1).standard_normal(50))
        result = await factory.discover("test", df, returns)
        assert result.success_count == 0
        factory.close()

    @pytest.mark.asyncio
    async def test_discover_llm_error(self):
        llm = _MockLLM(raise_error=True)
        factory = AlphaFactory(llm)
        df = _make_df(50)
        returns = pd.Series(np.random.default_rng(1).standard_normal(50))
        result = await factory.discover("test", df, returns)
        assert result.success_count == 0
        factory.close()

    @pytest.mark.asyncio
    async def test_discover_sandbox_fails(self):
        bad_factor = dict(_VALID_FACTOR)
        bad_factor["code"] = "import os\ndef compute(df, idx): return 0.0"
        # Refine also returns bad code
        llm = _MockLLM(response=[bad_factor])
        factory = AlphaFactory(llm, max_refine_attempts=0)
        df = _make_df(50)
        returns = pd.Series(np.random.default_rng(1).standard_normal(50))
        result = await factory.discover("test", df, returns)
        assert len(result.sandbox_failed) == 1
        assert result.success_count == 0
        factory.close()

    @pytest.mark.asyncio
    async def test_discover_full_pipeline(self):
        """End-to-end: good factor → sandbox → validation → store → registry."""
        # Create a factor that correlates with returns
        factor_code = (
            "def compute(df, idx):\n"
            "    if idx < 5:\n"
            "        return 0.0\n"
            "    return float(df['close'].iloc[idx] - df['close'].iloc[idx-5])\n"
        )
        good_factor = dict(_VALID_FACTOR)
        good_factor["code"] = factor_code
        good_factor["name"] = "momentum_5d_test"

        llm = _MockLLM(response=[good_factor])
        factory = AlphaFactory(
            llm,
            validation_thresholds=__import__(
                "alphaevo.alpha_factory.validator", fromlist=["ValidationThresholds"]
            ).ValidationThresholds(
                min_ic_abs=0.0,
                min_ir=0.0,
                min_monthly_win_rate=0.0,
            ),
        )

        np.random.default_rng(42)
        df = _make_df(200)
        factor_proxy = (df["close"] - df["close"].shift(5)).fillna(0.0)
        returns = pd.Series((factor_proxy * 0.01).values)

        result = await factory.discover("test", df, returns, register=True)
        assert len(result.proposed) == 1
        assert len(result.sandbox_passed) == 1

        # Check it was stored
        stored = factory.store.get("momentum_5d_test")
        assert stored is not None

        # Check it was registered (if validation passed)
        if result.success_count > 0:
            assert IndicatorRegistry.is_registered("momentum_5d_test")

        factory.close()

    @pytest.mark.asyncio
    async def test_store_access(self):
        llm = _MockLLM(response=[])
        factory = AlphaFactory(llm)
        assert factory.store is not None
        assert factory.store.count() == 0
        factory.close()

    @pytest.mark.asyncio
    async def test_process_candidate_respects_symbol_boundaries(self):
        llm = _MockLLM(response=[])
        factory = AlphaFactory(
            llm,
            validation_thresholds=__import__(
                "alphaevo.alpha_factory.validator", fromlist=["ValidationThresholds"]
            ).ValidationThresholds(
                min_ic_abs=0.0,
                min_ir=0.0,
                min_monthly_win_rate=0.0,
            ),
        )
        df_a = _make_df(40)
        df_b = _make_df(40)
        df_b["close"] = df_b["close"] + 1000
        combined = pd.concat(
            [
                df_a.assign(_symbol="AAA"),
                df_b.assign(_symbol="BBB"),
            ],
            ignore_index=True,
        )
        returns = combined.groupby("_symbol")["close"].pct_change().fillna(0.0)
        hyp = FactorHypothesis(
            name="boundary_safe_momentum",
            description="Uses previous close within one symbol only",
            rationale="Test grouped execution",
            code=(
                "def compute(df, idx):\n"
                "    if idx == 0:\n"
                "        return 0.0\n"
                "    return float(df['close'].iloc[idx] - df['close'].iloc[idx - 1])\n"
            ),
            expected_direction="positive",
        )
        result = DiscoveryResult()
        existing_factor_values: dict[str, pd.Series] = {}

        await factory._process_candidate(
            hyp,
            combined,
            returns,
            None,
            existing_factor_values,
            result,
            register=False,
        )

        factor_values = existing_factor_values["boundary_safe_momentum"]
        second_symbol_start = combined.index[combined["_symbol"] == "BBB"][0]
        assert factor_values.iloc[second_symbol_start] == 0.0
        factory.close()

    @pytest.mark.asyncio
    async def test_discover_passes_symbol_context_to_validator(self):
        llm = _MockLLM(response=[])
        factory = AlphaFactory(llm)
        captured: dict[str, object] = {}

        def fake_validate(*args, **kwargs):
            captured["symbols"] = kwargs.get("symbols")
            captured["expected_direction"] = kwargs.get("expected_direction")
            from alphaevo.alpha_factory.validator import ValidationResult

            return ValidationResult(factor_name="x", passed=True)

        factory._validator.validate = fake_validate  # type: ignore[method-assign]
        df = _make_df(20).assign(_symbol="AAA")
        hyp = FactorHypothesis(
            name="pass_symbols",
            description="test",
            rationale="test",
            code="def compute(df, idx):\n    return float(df['close'].iloc[idx])",
            expected_direction="positive",
        )
        result = DiscoveryResult()
        await factory._process_candidate(
            hyp,
            df,
            pd.Series(df["close"].pct_change().shift(-1).fillna(0).values),
            None,
            {},
            result,
            register=False,
        )

        assert captured["symbols"] is not None
        assert captured["expected_direction"] == "positive"
        factory.close()

    def test_load_registered_factors_restores_dynamic_registry(self, tmp_path):
        db_path = tmp_path / "alpha_factory.db"
        store = FactorStore(db_path)
        try:
            store.save(
                FactorRecord(
                    name="persisted_alpha_factor",
                    description="Persisted factor",
                    rationale="Testing reload",
                    code="def compute(df, idx):\n    return float(df['close'].iloc[idx])",
                    status="active",
                )
            )
        finally:
            store.close()

        loaded = load_registered_factors(db_path)

        assert "persisted_alpha_factor" in loaded
        assert IndicatorRegistry.is_registered("persisted_alpha_factor")
