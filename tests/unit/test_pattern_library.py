"""Tests for PatternLibrary — Voyager-inspired reusable strategy pattern store."""

import pytest

from alphaevo.models.enums import StrategyCategory
from alphaevo.models.execution import AntiFitMetrics, EvaluationReport, OverallMetrics
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    UniverseConfig,
)
from alphaevo.strategy.library import PatternLibrary, StrategyPattern


@pytest.fixture
def library():
    return PatternLibrary(db_path=":memory:")


@pytest.fixture
def sample_pattern():
    return StrategyPattern(
        pattern_id="entry_test_v1",
        name="RSI entry combo",
        category=StrategyCategory.REVERSAL,
        pattern_type="entry_combo",
        description="RSI oversold entry",
        conditions=[
            {"indicator": "rsi_14", "op": "<", "value": 30},
            {"indicator": "volume_ratio_1d_5d", "op": ">", "value": 1.5},
        ],
        source_strategy="test_v1",
        confidence_score=0.55,
        win_rate=0.60,
        signal_count=45,
    )


def _make_strategy(sid="test_v1") -> Strategy:
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name="Test Strategy",
            version=1,
            category=StrategyCategory.TREND,
        ),
        description="Test",
        universe=UniverseConfig(market=["us"]),
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
                StrategyCondition(indicator="ma5_above_ma10", op="==", value=True),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
        params=StrategyParams(),
    )


def _make_evaluation(sid="test_v1", score=0.55) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id=f"eval-{sid}",
        strategy_id=sid,
        overall=OverallMetrics(
            win_rate=0.60,
            avg_return=0.02,
            profit_loss_ratio=2.0,
            max_drawdown=0.08,
            sharpe_ratio=1.2,
            signal_count=50,
        ),
        confidence_score=score,
        anti_overfit=AntiFitMetrics(),
    )


class TestPatternLibrary:
    def test_save_and_get(self, library, sample_pattern):
        library.save(sample_pattern)
        retrieved = library.get("entry_test_v1")
        assert retrieved is not None
        assert retrieved.pattern_id == "entry_test_v1"
        assert retrieved.confidence_score == 0.55
        assert len(retrieved.conditions) == 2

    def test_get_nonexistent(self, library):
        assert library.get("not_a_real_id") is None

    def test_get_best_patterns(self, library):
        p1 = StrategyPattern(
            pattern_id="p1",
            name="P1",
            pattern_type="entry_combo",
            confidence_score=0.6,
            win_rate=0.55,
            signal_count=40,
        )
        p2 = StrategyPattern(
            pattern_id="p2",
            name="P2",
            pattern_type="entry_combo",
            confidence_score=0.4,
            win_rate=0.45,
            signal_count=30,
        )
        p3 = StrategyPattern(
            pattern_id="p3",
            name="P3",
            pattern_type="exit_config",
            confidence_score=0.7,
            win_rate=0.65,
            signal_count=55,
        )
        library.save(p1)
        library.save(p2)
        library.save(p3)

        # All above min_score=0.3
        all_patterns = library.get_best_patterns(min_score=0.3)
        assert len(all_patterns) == 3

        # Filter by type
        entries = library.get_best_patterns(pattern_type="entry_combo", min_score=0.3)
        assert len(entries) == 2

        # Filter by min_score
        high = library.get_best_patterns(min_score=0.5)
        assert len(high) == 2

    def test_record_usage(self, library, sample_pattern):
        library.save(sample_pattern)
        library.record_usage("entry_test_v1", succeeded=True)
        library.record_usage("entry_test_v1", succeeded=False)

        p = library.get("entry_test_v1")
        assert p.times_used == 2
        assert p.times_succeeded == 1
        assert p.success_rate == 0.5

    def test_extract_patterns_from_strategy(self, library):
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        patterns = library.extract_patterns_from_strategy(strategy, evaluation)

        # Should extract: entry_combo, exit_config, indicator_set
        assert len(patterns) == 3
        types = {p.pattern_type for p in patterns}
        assert "entry_combo" in types
        assert "exit_config" in types
        assert "indicator_set" in types

        # All should reference the source strategy
        for p in patterns:
            assert p.source_strategy == "test_v1"
            assert p.confidence_score == 0.55

    def test_extract_skips_low_quality(self, library):
        strategy = _make_strategy()
        low_eval = _make_evaluation(score=0.15)
        low_eval.overall.signal_count = 10
        patterns = library.extract_patterns_from_strategy(strategy, low_eval)
        assert len(patterns) == 0

    def test_format_for_prompt(self, library, sample_pattern):
        library.save(sample_pattern)
        text = library.format_for_prompt()
        assert "Successful Strategy Patterns" in text
        assert "RSI entry combo" in text

    def test_format_for_prompt_empty(self, library):
        assert library.format_for_prompt() == ""

    def test_upsert_pattern(self, library, sample_pattern):
        """Saving twice with same ID should update."""
        library.save(sample_pattern)
        sample_pattern.confidence_score = 0.80
        library.save(sample_pattern)
        p = library.get("entry_test_v1")
        assert p.confidence_score == 0.80

    def test_get_best_patterns_can_filter_by_source_family(self, library):
        family = StrategyPattern(
            pattern_id="family_p",
            name="Family Pattern",
            pattern_type="entry_combo",
            source_strategy="trend_pullback_rebound_v2",
            confidence_score=0.7,
        )
        foreign = StrategyPattern(
            pattern_id="foreign_p",
            name="Foreign Pattern",
            pattern_type="entry_combo",
            source_strategy="mean_reversion_oversold_v1",
            confidence_score=0.8,
        )
        library.save(family)
        library.save(foreign)

        filtered = library.get_best_patterns(source_family="trend_pullback_rebound", min_score=0.0)
        assert [p.pattern_id for p in filtered] == ["family_p"]

    def test_get_best_patterns_can_exclude_test_sources(self, library):
        test_pattern = StrategyPattern(
            pattern_id="test_p",
            name="Test Pattern",
            pattern_type="entry_combo",
            source_strategy="test_v1",
            confidence_score=0.9,
        )
        live_pattern = StrategyPattern(
            pattern_id="live_p",
            name="Live Pattern",
            pattern_type="entry_combo",
            source_strategy="ma_crossover_v2",
            confidence_score=0.8,
        )
        library.save(test_pattern)
        library.save(live_pattern)

        filtered = library.get_best_patterns(min_score=0.0, exclude_test_sources=True)

        assert [p.pattern_id for p in filtered] == ["live_p"]
