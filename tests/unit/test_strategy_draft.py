"""Tests for rule-based strategy drafting and revision."""

from __future__ import annotations

import pytest

from alphaevo.models.enums import StrategyCategory, StrategyStatus
from alphaevo.strategy.draft import StrategyDraftBuilder
from alphaevo.strategy.dsl.parser import StrategyParser


def test_draft_pullback_generates_valid_tunable_strategy() -> None:
    builder = StrategyDraftBuilder()

    strategy = builder.from_text(
        "趋势回踩10日线附近，放量反包，低回撤，止损3%，持有8天",
        strategy_id="trend_pullback_custom_v1",
    )

    StrategyParser().assert_valid(strategy)
    assert strategy.meta.id == "trend_pullback_custom_v1"
    assert strategy.meta.status == StrategyStatus.DRAFT
    assert strategy.meta.category == StrategyCategory.TREND
    assert strategy.exit.stop_loss.value == 0.03
    assert strategy.exit.max_holding_days == 8
    assert {condition.indicator for condition in strategy.entry.triggers} >= {
        "close_to_ma10_pct",
        "volume_ratio_1d_5d",
    }
    assert {param.target for param in strategy.params.tunable} >= {
        "entry.triggers[indicator=close_to_ma10_pct].value",
        "entry.triggers[indicator=volume_ratio_1d_5d].value",
        "exit.stop_loss.value",
    }


def test_draft_reversal_uses_reversal_conditions() -> None:
    strategy = StrategyDraftBuilder().from_text(
        "RSI超跌反转，止盈8%，美股短线",
        market="us",
    )

    StrategyParser().assert_valid(strategy)
    assert strategy.meta.category == StrategyCategory.REVERSAL
    assert strategy.universe.market == ["us"]
    assert strategy.exit.take_profit.type == "pct"
    assert strategy.exit.take_profit.value == 0.08
    assert [condition.indicator for condition in strategy.entry.triggers] == [
        "rsi_14",
        "deviation_from_ma20_pct",
        "volume_ratio_1d_5d",
    ]


def test_draft_extracts_explicit_ma_exit_before_holding_days() -> None:
    strategy = StrategyDraftBuilder().from_text(
        "RSI超跌反转，跌破10日线卖出，止损3%，持有5天",
        market="us",
    )

    StrategyParser().assert_valid(strategy)
    assert strategy.exit.max_holding_days == 5
    assert [condition.indicator for condition in strategy.exit.triggers] == ["close_below_ma10"]


def test_revise_tightens_links_parent_and_adds_confirmation() -> None:
    builder = StrategyDraftBuilder()
    base = builder.from_text(
        "趋势回踩放量策略",
        strategy_id="trend_pullback_v1",
    )
    original_stop = base.exit.stop_loss.value

    revised = builder.revise(
        base,
        "减少交易次数，降低回撤，右侧确认，持有5天",
    )

    StrategyParser().assert_valid(revised)
    assert revised.meta.id == "trend_pullback_v2"
    assert revised.meta.parent_id == "trend_pullback_v1"
    assert revised.meta.version == 2
    assert revised.exit.max_holding_days == 5
    assert revised.exit.stop_loss.value is not None
    assert original_stop is not None
    assert revised.exit.stop_loss.value < original_stop
    assert revised.entry.execution is not None
    assert revised.entry.execution.timing == "breakout_high"
    assert {condition.indicator for condition in revised.entry.triggers} >= {
        "momentum_10d",
        "body_to_range_ratio",
    }
    assert {condition.indicator for condition in revised.entry.guards} >= {"volatility_20d"}


def test_draft_rejects_short_selling_strategy() -> None:
    with pytest.raises(ValueError, match="Short-selling"):
        StrategyDraftBuilder().from_text("做空跌破20日线的股票")
