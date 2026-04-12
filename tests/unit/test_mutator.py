"""Tests for StrategyMutator — all change types, guardrails, version bumping."""

import pytest

from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import StrategyChange
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
from alphaevo.reflection.mutator import MutationError, StrategyMutator


def _make_strategy(
    strategy_id: str = "test_strat_v1",
    version: int = 1,
    conditions: list[StrategyCondition] | None = None,
    filters: list[StrategyCondition] | None = None,
) -> Strategy:
    """Create a minimal strategy for testing."""
    if conditions is None:
        conditions = [
            StrategyCondition(indicator="rsi_14", op="<", value=30),
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
        ]
    if filters is None:
        filters = [
            StrategyCondition(indicator="negative_news_score", op="<", value=0.4),
        ]
    return Strategy(
        meta=StrategyMeta(id=strategy_id, name="Test Strategy", version=version),
        description="Test strategy for mutation",
        universe=UniverseConfig(market=["us"]),
        entry=StrategyEntry(conditions=conditions, filters=filters),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.04),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
        params=StrategyParams(),
    )


@pytest.fixture
def mutator():
    return StrategyMutator(max_changes=3, complexity_limit=8)


# ── Version bumping ──────────────────────────────────────────────────


class TestVersionBumping:
    def test_basic_version_bump(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.meta.id == "test_strat_v2"
        assert new.meta.version == 2
        assert new.meta.parent_id == "test_strat_v1"

    def test_multi_version_bump(self, mutator):
        s = _make_strategy(strategy_id="my_strat_v3", version=3)
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                to_value=0.03,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.meta.id == "my_strat_v4"
        assert new.meta.version == 4
        assert new.meta.parent_id == "my_strat_v3"

    def test_original_unchanged(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
            )
        ]
        mutator.mutate(s, changes)
        assert s.meta.id == "test_strat_v1"
        assert s.meta.version == 1
        # Original condition unchanged
        rsi = [c for c in s.entry.conditions if c.indicator == "rsi_14"][0]
        assert rsi.value == 30


# ── Change types ─────────────────────────────────────────────────────


class TestChangeTypes:
    def test_tighten_filter(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
                reason="Tighter RSI threshold",
            )
        ]
        new = mutator.mutate(s, changes)
        rsi = [c for c in new.entry.conditions if c.indicator == "rsi_14"][0]
        assert rsi.value == 25

    def test_loosen_filter(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=35,
            )
        ]
        new = mutator.mutate(s, changes)
        rsi = [c for c in new.entry.conditions if c.indicator == "rsi_14"][0]
        assert rsi.value == 35

    def test_add_condition(self, mutator):
        s = _make_strategy()
        original_count = len(s.entry.conditions)
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                to_value={"indicator": "ma5_above_ma10", "op": "==", "value": True},
            )
        ]
        new = mutator.mutate(s, changes)
        assert len(new.entry.conditions) == original_count + 1
        added = [c for c in new.entry.conditions if c.indicator == "ma5_above_ma10"]
        assert len(added) == 1

    def test_add_filter(self, mutator):
        s = _make_strategy()
        original_count = len(s.entry.filters)
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.filters",
                to_value={"indicator": "st_flag", "op": "==", "value": False},
            )
        ]
        new = mutator.mutate(s, changes)
        assert len(new.entry.filters) == original_count + 1

    def test_remove_condition(self, mutator):
        s = _make_strategy()
        original_count = len(s.entry.conditions)
        changes = [
            StrategyChange(
                change_type=ChangeType.REMOVE_CONDITION,
                target="entry.conditions[indicator=rsi_14]",
            )
        ]
        new = mutator.mutate(s, changes)
        assert len(new.entry.conditions) == original_count - 1
        rsi = [c for c in new.entry.conditions if c.indicator == "rsi_14"]
        assert len(rsi) == 0

    def test_remove_filter(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.REMOVE_CONDITION,
                target="entry.conditions[indicator=negative_news_score]",
            )
        ]
        new = mutator.mutate(s, changes)
        news = [c for c in new.entry.filters if c.indicator == "negative_news_score"]
        assert len(news) == 0

    def test_adjust_exit_stop_loss(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                from_value=0.04,
                to_value=0.03,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.stop_loss.value == 0.03

    def test_adjust_exit_take_profit(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.take_profit.value",
                to_value=2.5,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.take_profit.value == 2.5

    def test_adjust_exit_take_profit_target_period(self, mutator):
        s = _make_strategy()
        s.exit.take_profit = TakeProfitConfig(type="target_ma", target="ma60")
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.take_profit.target",
                to_value=55,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.take_profit.target == "ma55"

    def test_adjust_exit_atr_stop_loss_period(self, mutator):
        s = _make_strategy()
        s.exit.stop_loss = StopLossConfig(type="atr", multiplier=2.0)
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.atr_period",
                to_value=21,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.stop_loss.atr_period == 21

    def test_adjust_exit_holding_days(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.max_holding_days",
                to_value=15,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.max_holding_days == 15

    def test_adjust_exit_type_normalizes_stop_loss_fields(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.type",
                to_value="price_level",
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.stop_loss.type == "price_level"
        assert new.exit.stop_loss.value is None
        assert new.exit.stop_loss.multiplier is None
        assert new.exit.stop_loss.atr_period is None

    def test_adjust_take_profit_type_normalizes_fields(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.take_profit.type",
                to_value="target_ma",
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.exit.take_profit.type == "target_ma"
        assert new.exit.take_profit.value is None
        assert new.exit.take_profit.trigger_pct is None
        assert new.exit.take_profit.trail_pct is None

    def test_change_universe(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.CHANGE_UNIVERSE,
                target="universe.market",
                to_value=["a_share_main"],
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.universe.market == ["a_share_main"]

    def test_tighten_filter_can_adjust_indicator_period(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="close_above_ma60", op="==", value=True)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=close_above_ma60].indicator",
                    range=[20, 120],
                    step=5,
                ),
                TunableParam(
                    target="entry.conditions[indicator=close_above_ma60].value",
                    range=[0, 1],
                    step=1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=close_above_ma60].indicator",
                to_value=55,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "close_above_ma55"
        assert (
            new.params.tunable[0].target == "entry.conditions[indicator=close_above_ma55].indicator"
        )
        assert new.params.tunable[1].target == "entry.conditions[indicator=close_above_ma55].value"

    def test_tighten_filter_can_adjust_window_indicator_period(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=volume_ratio_1d_5d].indicator",
                    range=[3, 20],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=volume_ratio_1d_5d].value",
                    range=[1.0, 3.0],
                    step=0.1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=volume_ratio_1d_5d].indicator",
                to_value=10,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "volume_ratio_1d_10d"
        assert (
            new.params.tunable[0].target
            == "entry.conditions[indicator=volume_ratio_1d_10d].indicator"
        )
        assert (
            new.params.tunable[1].target == "entry.conditions[indicator=volume_ratio_1d_10d].value"
        )

    def test_tighten_filter_can_adjust_atr_alias_period(self, mutator):
        s = _make_strategy(conditions=[StrategyCondition(indicator="atr", op=">", value=0.5)])
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=atr].indicator",
                    range=[7, 21],
                    step=1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=atr].indicator",
                to_value=21,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "atr_21"

    def test_tighten_filter_can_adjust_macd_signal_period(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="macd_histogram", op=">", value=0)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.fast",
                    range=[6, 18],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.slow",
                    range=[20, 40],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].indicator.signal",
                    range=[5, 15],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=macd_histogram].value",
                    range=[-1.0, 1.0],
                    step=0.1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=macd_histogram].indicator.signal",
                to_value=7,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "macd_histogram_fast12_slow26_signal7"
        assert (
            new.params.tunable[0].target
            == "entry.conditions[indicator=macd_histogram_fast12_slow26_signal7].indicator.fast"
        )
        assert (
            new.params.tunable[1].target
            == "entry.conditions[indicator=macd_histogram_fast12_slow26_signal7].indicator.slow"
        )
        assert (
            new.params.tunable[2].target
            == "entry.conditions[indicator=macd_histogram_fast12_slow26_signal7].indicator.signal"
        )
        assert (
            new.params.tunable[3].target
            == "entry.conditions[indicator=macd_histogram_fast12_slow26_signal7].value"
        )

    def test_tighten_filter_can_adjust_bollinger_std(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="bollinger_band_width", op="<", value=0.2)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width].indicator",
                    range=[10, 40],
                    step=5,
                ),
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width].indicator.std",
                    range=[1.0, 3.0],
                    step=0.5,
                ),
                TunableParam(
                    target="entry.conditions[indicator=bollinger_band_width].value",
                    range=[0.05, 0.5],
                    step=0.05,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=bollinger_band_width].indicator.std",
                to_value=2.5,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "bollinger_band_width_20d_std2p5"
        assert (
            new.params.tunable[0].target
            == "entry.conditions[indicator=bollinger_band_width_20d_std2p5].indicator"
        )
        assert (
            new.params.tunable[1].target
            == "entry.conditions[indicator=bollinger_band_width_20d_std2p5].indicator.std"
        )
        assert (
            new.params.tunable[2].target
            == "entry.conditions[indicator=bollinger_band_width_20d_std2p5].value"
        )

    def test_tighten_filter_can_adjust_dual_ma_fast_period(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                    range=[3, 8],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow",
                    range=[9, 20],
                    step=1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                to_value=6,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "ma6_ge_ma10_or_crossing"
        assert (
            new.params.tunable[0].target
            == "entry.conditions[indicator=ma6_ge_ma10_or_crossing].indicator.fast"
        )
        assert (
            new.params.tunable[1].target
            == "entry.conditions[indicator=ma6_ge_ma10_or_crossing].indicator.slow"
        )

    def test_tighten_filter_can_adjust_dual_ma_slow_period(self, mutator):
        s = _make_strategy(
            conditions=[StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True)]
        )
        s.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast",
                    range=[3, 8],
                    step=1,
                ),
                TunableParam(
                    target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow",
                    range=[9, 20],
                    step=1,
                ),
            ]
        )
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.slow",
                to_value=9,
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.conditions[0].indicator == "ma5_ge_ma9_or_crossing"


# ── Safety guardrails ────────────────────────────────────────────────


class TestGuardrails:
    def test_max_changes_enforced(self):
        mutator = StrategyMutator(max_changes=2)
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                to_value=0.03,
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.take_profit.value",
                to_value=3.0,
            ),
        ]
        new = mutator.mutate(s, changes)
        # 3rd change should be truncated
        assert new.exit.take_profit.value == 2.0  # unchanged

    def test_complexity_limit_trims(self):
        mutator = StrategyMutator(max_changes=10, complexity_limit=4)
        conditions = [StrategyCondition(indicator=f"ind_{i}", op=">", value=i) for i in range(3)]
        s = _make_strategy(conditions=conditions, filters=[])
        # Add 2 more conditions (using real registered indicators) to exceed limit of 4
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                to_value={"indicator": "rsi_14", "op": "<", "value": 70},
            ),
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                to_value={"indicator": "atr", "op": ">", "value": 0.5},
            ),
        ]
        new = mutator.mutate(s, changes)
        total = len(new.entry.conditions) + len(new.entry.filters)
        assert total <= 4

    def test_complexity_limit_trims_filters_when_dominant(self):
        """When filters alone exceed the limit, they must be trimmed too."""
        mutator = StrategyMutator(max_changes=10, complexity_limit=4)
        # 6 filters, 0 conditions — well over limit of 4
        filters = [StrategyCondition(indicator=f"filt_{i}", op=">", value=i) for i in range(6)]
        s = _make_strategy(conditions=[], filters=filters)
        # Trivial mutation that doesn't add complexity
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.filters[indicator=filt_0].value",
                to_value=10,
            ),
        ]
        new = mutator.mutate(s, changes)
        total = len(new.entry.conditions) + len(new.entry.filters)
        assert total <= 4
        # Filters should have been trimmed
        assert len(new.entry.filters) <= 4

    def test_no_valid_changes_raises(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=nonexistent].value",
                to_value=1,
            )
        ]
        with pytest.raises(MutationError, match="No changes could be applied"):
            mutator.mutate(s, changes)

    def test_add_condition_requires_dict(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                to_value="not a dict",
            )
        ]
        with pytest.raises(MutationError, match="No changes could be applied"):
            mutator.mutate(s, changes)


# ── Multiple changes ─────────────────────────────────────────────────


class TestMultipleChanges:
    def test_two_changes_applied(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
            ),
            StrategyChange(
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                to_value=0.03,
            ),
        ]
        new = mutator.mutate(s, changes)
        rsi = [c for c in new.entry.conditions if c.indicator == "rsi_14"][0]
        assert rsi.value == 25
        assert new.exit.stop_loss.value == 0.03
        assert new.meta.version == 2

    def test_atomic_bundle_rolls_back_on_failure(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value=25,
            ),
            StrategyChange(
                change_type=ChangeType.ADD_CONDITION,
                target="entry.conditions",
                to_value={"indicator": "missing_factor_xyz", "op": ">", "value": 1},
            ),
        ]

        with pytest.raises(MutationError, match="Atomic mutation failed"):
            mutator.mutate(s, changes, atomic=True)

        original_rsi = next(c for c in s.entry.conditions if c.indicator == "rsi_14")
        assert original_rsi.value == 30

    def test_atomic_bundle_rejects_complexity_trim(self):
        s = _make_strategy(
            conditions=[
                StrategyCondition(indicator=f"rsi_{i}", op="<", value=30) for i in range(6)
            ],
            filters=[
                StrategyCondition(indicator=f"filter_{i}", op=">", value=0.5) for i in range(3)
            ],
        )
        mutator = StrategyMutator(max_changes=3, complexity_limit=8)
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_0].value",
                to_value=25,
            )
        ]

        with pytest.raises(MutationError, match="Atomic mutation would exceed complexity limit"):
            mutator.mutate(s, changes, atomic=True)


# ── CHANGE_LOGIC ─────────────────────────────────────────────────────


class TestChangeLogic:
    def test_and_to_or(self, mutator):
        s = _make_strategy()
        assert s.entry.logic == "and"
        changes = [
            StrategyChange(
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                from_value="and",
                to_value="or",
                reason="test logic switch",
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.logic == "or"

    def test_or_to_and(self, mutator):
        s = _make_strategy()
        s.entry.logic = "or"
        changes = [
            StrategyChange(
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                from_value="or",
                to_value="and",
                reason="test logic switch back",
            )
        ]
        new = mutator.mutate(s, changes)
        assert new.entry.logic == "and"

    def test_invalid_logic_value_rejected(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.CHANGE_LOGIC,
                target="entry.logic",
                to_value="xor",
            )
        ]
        with pytest.raises(MutationError, match="No changes could be applied"):
            mutator.mutate(s, changes)


# ── DISCOVER_FACTOR ──────────────────────────────────────────────────


class TestDiscoverFactor:
    def test_discover_factor_adds_condition(self, mutator):
        """New factor is added as a condition when registered."""
        from alphaevo.backtest.indicators import IndicatorRegistry

        # Register a temporary dynamic indicator
        IndicatorRegistry.register_dynamic(
            "_test_discovered_factor",
            lambda df, idx, ctx=None: 1.0,
        )
        try:
            s = _make_strategy()
            initial_count = len(s.entry.conditions)
            changes = [
                StrategyChange(
                    change_type=ChangeType.DISCOVER_FACTOR,
                    target="entry.conditions",
                    to_value={
                        "indicator": "_test_discovered_factor",
                        "op": ">",
                        "value": 0.5,
                    },
                    reason="AlphaFactory discovered factor",
                )
            ]
            new = mutator.mutate(s, changes)
            assert len(new.entry.conditions) == initial_count + 1
            added = new.entry.conditions[-1]
            assert added.indicator == "_test_discovered_factor"
            assert added.op == ">"
            assert added.value == 0.5
        finally:
            IndicatorRegistry._dynamic_registry.pop("_test_discovered_factor", None)

    def test_discover_factor_unregistered_rejected(self, mutator):
        s = _make_strategy()
        changes = [
            StrategyChange(
                change_type=ChangeType.DISCOVER_FACTOR,
                target="entry.conditions",
                to_value={
                    "indicator": "_nonexistent_factor_xyz",
                    "op": ">",
                    "value": 0.0,
                },
            )
        ]
        with pytest.raises(MutationError, match="No changes could be applied"):
            mutator.mutate(s, changes)


class TestComplexityTrimPriority:
    """N-05: Verify conditions are trimmed before filters."""

    def test_conditions_trimmed_first(self):
        """When total exceeds limit, conditions should be trimmed before filters."""
        conditions = [
            StrategyCondition(indicator=f"rsi_{i}", op="<", value=30) for i in range(6)
        ]
        filters = [
            StrategyCondition(indicator=f"filter_{i}", op=">", value=0.5) for i in range(4)
        ]
        s = _make_strategy(conditions=conditions, filters=filters)
        mutator = StrategyMutator(complexity_limit=8)
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_0].value",
                to_value=25,
            )
        ]
        result = mutator.mutate(s, changes)
        total = len(result.entry.conditions) + len(result.entry.filters)
        assert total <= 8
        # All 6 conditions preserved (priority)
        assert len(result.entry.conditions) == 6
        # Filters trimmed to remaining budget: 8 - 6 = 2
        assert len(result.entry.filters) == 2

    def test_filters_trimmed_when_conditions_alone_exceed(self):
        """When conditions alone exceed limit, all filters are dropped."""
        conditions = [
            StrategyCondition(indicator=f"rsi_{i}", op="<", value=30) for i in range(10)
        ]
        filters = [
            StrategyCondition(indicator=f"filter_{i}", op=">", value=0.5) for i in range(3)
        ]
        s = _make_strategy(conditions=conditions, filters=filters)
        mutator = StrategyMutator(complexity_limit=8)
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_0].value",
                to_value=25,
            )
        ]
        result = mutator.mutate(s, changes)
        assert len(result.entry.conditions) == 8
        assert len(result.entry.filters) == 0


# ── _sanitize_condition_value tests ──────────────────────────────────


class TestSanitizeConditionValue:
    """Test that LLM-output values with operator prefixes are cleaned."""

    def test_strips_less_than_operator(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("< 65.0") == 65.0

    def test_strips_less_than_equals(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("<= 0.015") == 0.015

    def test_strips_greater_than(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("> 1.5") == 1.5

    def test_strips_greater_than_equals(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value(">= 30") == 30

    def test_strips_double_equals(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("== true") is True

    def test_strips_not_equals(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("!= 0") == 0

    def test_passthrough_float(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value(65.0) == 65.0

    def test_passthrough_int(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value(30) == 30

    def test_passthrough_bool(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value(True) is True

    def test_bool_string_true(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("true") is True

    def test_bool_string_false(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("false") is False

    def test_numeric_string(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("65.0") == 65.0

    def test_integer_string(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("30") == 30

    def test_non_numeric_passthrough(self):
        from alphaevo.reflection.mutator import _sanitize_condition_value

        assert _sanitize_condition_value("some_text") == "some_text"


class TestMutatorWithLLMValues:
    """End-to-end: verify mutator handles LLM-formatted values correctly."""

    def test_tighten_with_operator_prefix(self):
        """LLM sends '< 65.0' for RSI filter — should set cond.value to 65.0."""
        s = _make_strategy()
        mutator = StrategyMutator()
        changes = [
            StrategyChange(
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value="< 65.0",
            )
        ]
        result = mutator.mutate(s, changes)
        rsi_cond = next(c for c in result.entry.conditions if c.indicator == "rsi_14")
        assert rsi_cond.value == 65.0
        assert isinstance(rsi_cond.value, float)

    def test_loosen_with_operator_prefix(self):
        s = _make_strategy()
        mutator = StrategyMutator()
        changes = [
            StrategyChange(
                change_type=ChangeType.LOOSEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                to_value="> 35",
            )
        ]
        result = mutator.mutate(s, changes)
        rsi_cond = next(c for c in result.entry.conditions if c.indicator == "rsi_14")
        assert rsi_cond.value == 35
