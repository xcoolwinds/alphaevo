"""Tests for factor CLI helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from alphaevo.cli.commands.factor import _load_factor_history, _render_factor_discovery_report


@pytest.mark.asyncio
async def test_load_factor_history_uses_data_manager_with_cache(tmp_path: Path):
    config = SimpleNamespace(
        data=SimpleNamespace(
            adapter="yfinance",
            cache_dir=tmp_path / "cache",
            dsa_path=None,
        )
    )
    history = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5, freq="B")})

    with (
        patch("alphaevo.data.adapter.get_adapter", return_value=MagicMock()) as mock_get_adapter,
        patch("alphaevo.data.adapter.DataManager") as mock_data_manager,
    ):
        manager = mock_data_manager.return_value
        manager.get_history = AsyncMock(return_value=history)

        result = await _load_factor_history(config, "AAPL", days=120)

    assert result is history
    mock_get_adapter.assert_called_once_with(config)
    cache = mock_data_manager.call_args.kwargs["cache"]
    assert cache.cache_dir == config.data.cache_dir
    fetch_call = manager.get_history.await_args
    assert fetch_call.args[0] == "AAPL"
    assert (fetch_call.args[2] - fetch_call.args[1]).days == 119


def test_render_factor_discovery_report_includes_registered_metrics(tmp_path: Path) -> None:
    from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore

    store = FactorStore(tmp_path / "factors.db")
    try:
        store.save(
            FactorRecord(
                name="alpha_volume_spike",
                description="Volume expansion into breakouts",
                rationale="Confirms breakouts when participation rises",
                code="def compute(df, idx): return 0.0",
                ic_mean=0.12,
                ir=0.65,
                monthly_win_rate=0.75,
                turnover=0.22,
            )
        )
        result = SimpleNamespace(
            proposed=[
                SimpleNamespace(
                    name="alpha_volume_spike",
                    description="Volume expansion into breakouts",
                    rationale="Confirms breakouts when participation rises",
                    expected_direction="positive",
                )
            ],
            sandbox_passed=["alpha_volume_spike"],
            sandbox_failed=[],
            validation_passed=["alpha_volume_spike"],
            validation_failed=[],
            registered=["alpha_volume_spike"],
        )
        report = _render_factor_discovery_report(
            "AAPL",
            "find false breakout filters",
            result,
            factor_store=store,
        )
    finally:
        store.close()

    assert "# Factor Discovery Report" in report
    assert "## LLM Proposals" in report
    assert "## Registered Factors" in report
    assert "IC Mean: 0.120" in report
    assert "find false breakout filters" in report
