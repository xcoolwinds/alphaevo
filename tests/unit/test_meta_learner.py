"""Tests for MetaLearner — learns optimal evolution strategies from experience."""

import pytest

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import AntiFitMetrics, EvaluationReport, OverallMetrics
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.reflection.meta_learner import EvolutionProfile, MetaInsight, MetaLearner


@pytest.fixture
def store():
    return ExperienceStore(db_path=":memory:")


@pytest.fixture
def populated_store(store):
    """Store with some evolution history."""
    records = [
        ExperienceRecord(
            strategy_family="trend",
            strategy_id="trend_v1",
            round_num=1,
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value="30",
            to_value="25",
            reason="Tighten RSI",
            score_before=0.30,
            score_after=0.40,
            score_delta=0.10,
            worked=True,
            lesson="Tightening RSI improved win rate",
        ),
        ExperienceRecord(
            strategy_family="trend",
            strategy_id="trend_v2",
            round_num=2,
            change_type=ChangeType.ADJUST_EXIT,
            target="exit.stop_loss.value",
            from_value="0.04",
            to_value="0.03",
            reason="Tighten stop loss",
            score_before=0.40,
            score_after=0.45,
            score_delta=0.05,
            worked=True,
            lesson="Tighter stop loss helped",
        ),
        ExperienceRecord(
            strategy_family="trend",
            strategy_id="trend_v3",
            round_num=3,
            change_type=ChangeType.LOOSEN_FILTER,
            target="entry.conditions[indicator=volume_ratio_1d_5d].value",
            from_value="1.5",
            to_value="1.2",
            reason="Allow more signals",
            score_before=0.45,
            score_after=0.42,
            score_delta=-0.03,
            worked=False,
            lesson="Loosening volume filter did not help",
        ),
        ExperienceRecord(
            strategy_family="trend",
            strategy_id="trend_v4",
            round_num=4,
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value="25",
            to_value="22",
            reason="Further tighten RSI",
            score_before=0.42,
            score_after=0.43,
            score_delta=0.01,
            worked=True,
            lesson="Small improvement from RSI tightening",
        ),
    ]
    store.record_batch(records)
    return store


class TestMetaLearner:
    def test_empty_store_returns_defaults(self, store):
        learner = MetaLearner(store)
        profile = learner.analyze()
        assert isinstance(profile, EvolutionProfile)
        assert profile.recommended_intensity == 1.0
        assert len(profile.insights) == 4  # 4 analysis dimensions

    def test_analyze_with_history(self, populated_store):
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="trend")
        assert isinstance(profile, EvolutionProfile)
        assert len(profile.insights) == 4

        # Should have some change effectiveness data
        eff = next(i for i in profile.insights if i.insight_type == "change_effectiveness")
        assert eff.description != "No data available"

    def test_analyze_family_filter(self, populated_store):
        """Querying unknown family should still return defaults."""
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="nonexistent_family")
        assert isinstance(profile, EvolutionProfile)

    def test_insights_have_required_fields(self, populated_store):
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="trend")
        for insight in profile.insights:
            assert isinstance(insight, MetaInsight)
            assert insight.insight_type in (
                "change_effectiveness",
                "problem_pattern",
                "optimal_intensity",
                "convergence",
            )
            assert isinstance(insight.confidence, float)
            assert isinstance(insight.data, dict)

    def test_convergence_estimation(self, populated_store):
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="trend")
        conv = next(i for i in profile.insights if i.insight_type == "convergence")
        estimated = conv.data.get("estimated_rounds", 5)
        assert 3 <= estimated <= 10

    def test_optimal_intensity(self, populated_store):
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="trend")
        intensity = next(i for i in profile.insights if i.insight_type == "optimal_intensity")
        recommended = intensity.data.get("recommended", 1.0)
        assert 0.5 <= recommended <= 2.0

    def test_change_effectiveness_respects_family_scope(self, store):
        store.record_batch(
            [
                ExperienceRecord(
                    strategy_family="trend",
                    strategy_id="trend_v1",
                    round_num=1,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value="30",
                    to_value="25",
                    reason="trend success",
                    score_before=0.30,
                    score_after=0.40,
                    score_delta=0.10,
                    worked=True,
                    lesson="trend success",
                ),
                ExperienceRecord(
                    strategy_family="reversal",
                    strategy_id="reversal_v1",
                    round_num=1,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value="30",
                    to_value="25",
                    reason="reversal failure",
                    score_before=0.30,
                    score_after=0.20,
                    score_delta=-0.10,
                    worked=False,
                    lesson="reversal failure",
                ),
            ]
        )

        learner = MetaLearner(store)
        profile = learner.analyze(family_id="trend")
        insight = next(i for i in profile.insights if i.insight_type == "change_effectiveness")

        rates = insight.data["rates"]
        assert rates[ChangeType.TIGHTEN_FILTER.value]["rate"] == pytest.approx(1.0)

    def test_unknown_family_reports_family_specific_empty_history(self, populated_store):
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="nonexistent_family")
        insight = next(i for i in profile.insights if i.insight_type == "change_effectiveness")
        # With cold start, unknown family gets playbook priors instead of empty msg
        assert "cold start" in insight.description.lower()
        assert insight.data.get("ranked")

    def test_evaluation_can_raise_recommended_intensity(self, populated_store):
        learner = MetaLearner(populated_store)
        evaluation = EvaluationReport(
            strategy_id="trend_v1",
            overall=OverallMetrics(
                win_rate=0.45,
                avg_return=0.01,
                profit_loss_ratio=1.2,
                max_drawdown=0.10,
                sharpe_ratio=0.7,
                signal_count=8,
            ),
            anti_overfit=AntiFitMetrics(),
            confidence_score=0.2,
        )

        profile = learner.analyze(family_id="trend", evaluation=evaluation)
        assert profile.recommended_intensity >= 1.5
        assert profile.recommended_method == "hybrid"

    def test_low_signal_reduces_max_changes(self, populated_store):
        """When signal_count is very low, meta should cap max_changes to 2."""
        learner = MetaLearner(populated_store)
        evaluation = EvaluationReport(
            strategy_id="trend_v1",
            overall=OverallMetrics(
                win_rate=0.40,
                avg_return=0.01,
                profit_loss_ratio=1.2,
                max_drawdown=0.10,
                sharpe_ratio=0.7,
                signal_count=10,
            ),
            anti_overfit=AntiFitMetrics(),
            confidence_score=0.2,
        )
        profile = learner.analyze(family_id="trend", evaluation=evaluation)
        assert profile.recommended_max_changes <= 2

    def test_high_win_rate_reduces_max_changes(self, populated_store):
        """Strong strategies (≥ 60% WR, ≥ 50 signals) should fine-tune with few changes."""
        learner = MetaLearner(populated_store)
        evaluation = EvaluationReport(
            strategy_id="trend_v1",
            overall=OverallMetrics(
                win_rate=0.65,
                avg_return=0.03,
                profit_loss_ratio=2.5,
                max_drawdown=0.08,
                sharpe_ratio=1.5,
                signal_count=60,
            ),
            anti_overfit=AntiFitMetrics(),
            confidence_score=0.7,
        )
        profile = learner.analyze(family_id="trend", evaluation=evaluation)
        assert profile.recommended_max_changes <= 2
        assert profile.recommended_method == "param_search"

    def test_many_bad_signals_allows_more_changes(self, populated_store):
        """Many bad signals (>100, WR<40%) should allow more aggressive changes."""
        learner = MetaLearner(populated_store)
        evaluation = EvaluationReport(
            strategy_id="trend_v1",
            overall=OverallMetrics(
                win_rate=0.30,
                avg_return=-0.01,
                profit_loss_ratio=0.8,
                max_drawdown=0.20,
                sharpe_ratio=0.3,
                signal_count=150,
            ),
            anti_overfit=AntiFitMetrics(),
            confidence_score=0.15,
        )
        profile = learner.analyze(family_id="trend", evaluation=evaluation)
        # Should allow more changes than default (2) but respect config cap (3)
        assert profile.recommended_max_changes == 3


class TestMetaLearnerColdStart:
    """Cold start: when ExperienceStore is empty, use builtin playbook priors."""

    def test_empty_store_cold_start_returns_priors(self, store):
        """With zero experience, change_effectiveness should use playbook priors."""
        learner = MetaLearner(store)
        profile = learner.analyze()
        eff = next(i for i in profile.insights if i.insight_type == "change_effectiveness")
        # Should have data from builtin playbooks, not "No data available"
        assert "cold start" in eff.description.lower() or "builtin" in eff.description.lower()
        assert eff.confidence > 0
        assert eff.data.get("ranked")

    def test_cold_start_populates_preferred_change_types(self, store):
        """Cold start priors should populate preferred_change_types."""
        learner = MetaLearner(store)
        profile = learner.analyze()
        # The _build_recommendations method should use ranked from cold start
        assert len(profile.preferred_change_types) > 0

    def test_cold_start_family_specific_also_works(self, store):
        """Even with a family_id but no records, cold start should activate."""
        learner = MetaLearner(store)
        profile = learner.analyze(family_id="nonexistent")
        eff = next(i for i in profile.insights if i.insight_type == "change_effectiveness")
        assert eff.data.get("ranked")

    def test_populated_store_skips_cold_start(self, populated_store):
        """When experience exists, real data is used instead of cold start."""
        learner = MetaLearner(populated_store)
        profile = learner.analyze(family_id="trend")
        eff = next(i for i in profile.insights if i.insight_type == "change_effectiveness")
        # Real data should have "Best:" in description, not "cold start"
        assert "cold start" not in eff.description.lower()
