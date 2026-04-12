"""Tests for SelfCritic — validates proposed strategy changes before mutation."""

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    AntiFitMetrics,
    EvaluationReport,
    OverallMetrics,
    ReflectionResult,
    StrategyChange,
)
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
    UniverseConfig,
)
from alphaevo.reflection.critic import CritiqueVerdict, SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(id="test_v1", name="Test", version=1),
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
        params=StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=rsi_14].value",
                    range=[20.0, 40.0],
                    step=5.0,
                ),
            ]
        ),
    )


def _make_evaluation(win_rate=0.50, signal_count=50) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-test",
        strategy_id="test_v1",
        overall=OverallMetrics(
            win_rate=win_rate,
            avg_return=0.01,
            profit_loss_ratio=1.5,
            max_drawdown=0.10,
            sharpe_ratio=0.8,
            signal_count=signal_count,
        ),
        confidence_score=0.4,
        anti_overfit=AntiFitMetrics(),
    )


def _make_reflection(changes: list[StrategyChange]) -> ReflectionResult:
    return ReflectionResult(
        strategy_id="test_v1",
        evaluation_id="eval-test",
        failure_patterns=["low win rate"],
        proposed_changes=changes,
        reflection_summary="Test reflection",
    )


class TestSelfCritic:
    def test_approve_valid_changes(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="Tighten RSI threshold",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.approved) == 1
        assert len(verdict.rejected) == 0

    def test_reject_unknown_indicator(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                from_value=None,
                to_value={"indicator": "totally_fake_indicator", "op": ">", "value": 0.5},
                reason="Add fake indicator",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "Unknown indicator" in verdict.rejected[0][1]

    def test_reject_exceeding_complexity(self):
        """Adding too many conditions should be rejected."""
        critic = SelfCritic(complexity_limit=3)
        strategy = _make_strategy()  # already has 2 conditions
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                from_value=None,
                to_value={"indicator": "rsi_14", "op": ">", "value": 50},
                reason="Add condition 1",
            ),
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                from_value=None,
                to_value={"indicator": "ma5_above_ma10", "op": "==", "value": True},
                reason="Add condition 2",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        # 2 existing + 2 new = 4 > limit of 3; both should be rejected
        assert len(verdict.rejected) == 2

    def test_reject_tight_stop_loss(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                from_value=0.04,
                to_value=0.003,
                reason="Tighten stop loss",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "too tight" in verdict.rejected[0][1].lower()

    def test_reject_wide_stop_loss(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                from_value=0.04,
                to_value=0.25,
                reason="Widen stop loss",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "too wide" in verdict.rejected[0][1].lower()

    def test_reject_remove_condition_when_low_win_rate(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.30)
        changes = [
            StrategyChange(
                change_type=ChangeType.REMOVE_CONDITION,
                target="entry.conditions[indicator=rsi_14]",
                from_value=None,
                to_value=None,
                reason="Remove RSI filter",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1

    def test_allow_remove_condition_when_low_signal_strategy_is_stuck(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.0, signal_count=0)
        changes = [
            StrategyChange(
                change_type=ChangeType.REMOVE_CONDITION,
                target="entry.conditions[indicator=rsi_14]",
                from_value=None,
                to_value=None,
                reason="Remove blocking RSI filter",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.approved) == 1
        assert len(verdict.rejected) == 0

    def test_reject_add_condition_when_signal_count_is_very_low(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.0, signal_count=0)
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                from_value=None,
                to_value={"indicator": "rsi_14", "op": ">", "value": 50},
                reason="Add another filter",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "signal_count is very low" in verdict.rejected[0][1]

    def test_consistency_check_removes_conflicting(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="Tighten RSI",
            ),
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=35,
                reason="Loosen RSI",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        # Should warn about contradiction
        assert len(verdict.warnings) > 0
        assert any("ontradictory" in w or "onflicting" in w.lower() for w in verdict.warnings)

    def test_verdict_approval_rate(self):
        verdict = CritiqueVerdict()
        assert verdict.approval_rate == 1.0  # empty = 100%

        # Add approved and rejected
        change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="test",
            from_value=1,
            to_value=2,
            reason="test",
        )
        verdict.approved.append(change)
        verdict.rejected.append((change, "reason"))
        assert verdict.approval_rate == 0.5

    def test_empty_changes_returns_empty_verdict(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        reflection = _make_reflection([])
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.approved) == 0
        assert len(verdict.rejected) == 0

    def test_value_out_of_tunable_range(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=50,  # range is [20, 40], so 50 is out
                reason="Set RSI too high",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "outside" in verdict.rejected[0][1].lower()

    def test_valid_atr_exit_bundle_is_approved(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.type",
                from_value="pct",
                to_value="atr",
                reason="adapt to volatility",
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.multiplier",
                from_value=None,
                to_value=2.5,
                reason="set atr multiple",
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.atr_period",
                from_value=None,
                to_value=21,
                reason="use slower ATR",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 0
        assert len(verdict.approved) == 3

    def test_invalid_price_level_exit_bundle_is_rejected(self):
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.type",
                from_value="pct",
                to_value="price_level",
                reason="switch to structural stop",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "price_level stop loss requires value or reference" in verdict.rejected[0][1]

    def test_rejects_direct_reversal_of_recent_success(self):
        store = ExperienceStore(":memory:")
        store.record(
            ExperienceRecord(
                strategy_family="test",
                strategy_id="test_v2",
                round_num=1,
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                from_value="and",
                to_value="or",
                reason="Unlock sparse strategy",
                score_before=0.081,
                score_after=0.377,
                score_delta=0.296,
                worked=True,
            )
        )
        critic = SelfCritic(experience_store=store)
        strategy = _make_strategy()
        strategy.meta.id = "test_v2"
        strategy.meta.parent_id = "test_v1"
        strategy.entry.logic = "or"
        evaluation = _make_evaluation(win_rate=0.52, signal_count=500)
        changes = [
            StrategyChange(
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                from_value="or",
                to_value="and",
                reason="Undo the previous switch",
            ),
        ]
        reflection = _make_reflection(changes)
        verdict = critic.critique(strategy, evaluation, reflection)
        assert len(verdict.rejected) == 1
        assert "recently successful change" in verdict.rejected[0][1]


class TestCriticDynamicResolution:
    """N-06: Verify tighten/loosen conflict resolution depends on win_rate."""

    def _conflicting_changes(self) -> list[StrategyChange]:
        return [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="Tighten RSI",
            ),
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=35,
                reason="Loosen RSI",
            ),
        ]

    def test_low_win_rate_keeps_tighten(self):
        """Low win_rate should prefer tighten to improve quality."""
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.40)
        reflection = _make_reflection(self._conflicting_changes())
        verdict = critic.critique(strategy, evaluation, reflection)
        types = [c.change_type for c in verdict.approved]
        assert ChangeType.TIGHTEN_FILTER in types
        assert ChangeType.LOOSEN_FILTER not in types
        assert any("tighten" in w.lower() for w in verdict.warnings)

    def test_high_win_rate_keeps_loosen(self):
        """High win_rate (>=55%) should prefer loosen to get more signals."""
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.60)
        reflection = _make_reflection(self._conflicting_changes())
        verdict = critic.critique(strategy, evaluation, reflection)
        types = [c.change_type for c in verdict.approved]
        assert ChangeType.LOOSEN_FILTER in types
        assert ChangeType.TIGHTEN_FILTER not in types
        assert any("loosen" in w.lower() for w in verdict.warnings)

    def test_sparse_signal_conflict_keeps_loosen_even_with_low_win_rate(self):
        """Sparse strategies should prefer loosen to recover enough signals."""
        critic = SelfCritic()
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.20, signal_count=5)
        reflection = _make_reflection(self._conflicting_changes())
        verdict = critic.critique(strategy, evaluation, reflection)
        types = [c.change_type for c in verdict.approved]
        assert ChangeType.LOOSEN_FILTER in types
        assert ChangeType.TIGHTEN_FILTER not in types
        assert any("loosen" in w.lower() for w in verdict.warnings)
