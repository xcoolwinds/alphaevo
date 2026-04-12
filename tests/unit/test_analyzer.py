"""Tests for ReflectionAnalyzer — mocked LLM reflection + heuristic fallback."""

import json
from datetime import date as date_type

import pytest

from alphaevo.core.config import LLMConfig
from alphaevo.core.llm import LLMClient
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    EvaluationReport,
    OverallMetrics,
    TradeSignal,
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
from alphaevo.reflection.analyzer import ReflectionAnalyzer


def _make_strategy() -> Strategy:
    return Strategy(
        meta=StrategyMeta(id="test_v1", name="Test", version=1),
        description="Test strategy",
        universe=UniverseConfig(market=["us"]),
        entry=StrategyEntry(
            conditions=[
                StrategyCondition(indicator="rsi_14", op="<", value=30),
                StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
            ],
        ),
        exit=StrategyExit(
            stop_loss=StopLossConfig(type="pct", value=0.05),
            take_profit=TakeProfitConfig(type="rr", value=2.0),
            max_holding_days=10,
        ),
        params=StrategyParams(),
    )


def _make_evaluation(
    win_rate: float = 0.45,
    avg_return: float = 0.01,
    pl_ratio: float = 1.2,
    max_drawdown: float = 0.10,
    sharpe: float = 0.8,
    signal_count: int = 50,
    confidence: float = 0.35,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-001",
        strategy_id="test_v1",
        overall=OverallMetrics(
            win_rate=win_rate,
            avg_return=avg_return,
            profit_loss_ratio=pl_ratio,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe,
            signal_count=signal_count,
        ),
        failure_cases=[
            TradeSignal(
                symbol="AAPL",
                signal_date=date_type(2024, 1, 15),
                direction="long",
                entry_price=180.0,
                exit_price=170.0,
                return_pct=-0.0556,
                exit_reason="stop_loss",
            ),
        ],
        confidence_score=confidence,
    )


def _mock_llm_response(llm, response_data: dict | list[dict]):
    """Mock the LLM to return JSON dict(s).

    If *response_data* is a list, each LLM call gets the next item in order.
    If it's a single dict, every call returns the same response.
    """
    if isinstance(response_data, dict):
        responses = [response_data]
        cycle = True
    else:
        responses = list(response_data)
        cycle = False

    call_idx = [0]

    class FakeLitellm:
        @staticmethod
        def completion(**kwargs):
            idx = call_idx[0]
            if cycle:
                data = responses[0]
            else:
                data = responses[min(idx, len(responses) - 1)]
            call_idx[0] += 1

            class FakeMsg:
                content = json.dumps(data)

            class FakeChoice:
                message = FakeMsg()

            class FakeResp:
                choices = [FakeChoice()]

            return FakeResp()

    llm._litellm = FakeLitellm()


def _mock_two_step_response(llm, diagnosis_data: dict, experiment_data: dict):
    """Mock LLM for two-step reflection: first call returns diagnosis, second returns experiments."""
    _mock_llm_response(llm, [diagnosis_data, experiment_data])


@pytest.fixture
def llm():
    config = LLMConfig(model="test-model")
    return LLMClient(config)


# ── LLM-based reflection ────────────────────────────────────────────


class TestLLMReflection:
    def test_basic_reflection(self, llm):
        _mock_two_step_response(
            llm,
            # Step 1: diagnosis
            {
                "root_causes": [
                    {"problem": "Low win rate", "evidence": "42% < 50%", "severity": "high"},
                    {
                        "problem": "Poor risk/reward",
                        "evidence": "PL ratio 1.2",
                        "severity": "medium",
                    },
                ],
                "diagnosis_summary": "Strategy needs tighter entry and wider targets",
                "structural_issues": [],
            },
            # Step 2: experiment design
            {
                "candidates": [
                    {
                        "hypothesis": {
                            "problem": "RSI threshold too loose",
                            "hypothesis": "Tightening RSI will filter out false signals",
                            "expected_outcome": "+5% win rate",
                            "confidence": 0.7,
                        },
                        "proposed_changes": [
                            {
                                "change_type": "tighten_filter",
                                "target": "entry.conditions[indicator=rsi_14].value",
                                "from_value": 30,
                                "to_value": 25,
                                "reason": "Tighter RSI threshold",
                            }
                        ],
                        "priority_score": 0.8,
                        "rationale": "Most likely to improve win rate",
                    }
                ],
            },
        )
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        strategy = _make_strategy()
        evaluation = _make_evaluation()

        result = analyzer.reflect(strategy, evaluation)

        assert result.strategy_id == "test_v1"
        assert result.evaluation_id == "eval-001"
        assert len(result.failure_patterns) == 2
        assert len(result.proposed_changes) == 1
        assert result.proposed_changes[0].change_type == ChangeType.TIGHTEN_FILTER
        assert result.proposed_changes[0].to_value == 25

    def test_max_changes_enforced(self, llm):
        many_changes = [
            {
                "change_type": "tighten_filter",
                "target": f"entry.conditions[indicator=ind_{i}].value",
                "to_value": i,
                "reason": f"Change {i}",
            }
            for i in range(5)
        ]
        _mock_two_step_response(
            llm,
            {"root_causes": [], "diagnosis_summary": "Many issues", "structural_issues": []},
            {
                "candidates": [
                    {
                        "hypothesis": {
                            "problem": "test",
                            "hypothesis": "test",
                            "expected_outcome": "test",
                            "confidence": 0.5,
                        },
                        "proposed_changes": many_changes,
                        "priority_score": 0.8,
                        "rationale": "test",
                    }
                ],
            },
        )
        analyzer = ReflectionAnalyzer(llm, max_changes=2)
        result = analyzer.reflect(_make_strategy(), _make_evaluation())
        assert len(result.proposed_changes) <= 2

    def test_unknown_change_type_skipped(self, llm):
        _mock_two_step_response(
            llm,
            {"root_causes": [], "diagnosis_summary": "test", "structural_issues": []},
            {
                "candidates": [
                    {
                        "hypothesis": {
                            "problem": "test",
                            "hypothesis": "test",
                            "expected_outcome": "",
                            "confidence": 0.5,
                        },
                        "proposed_changes": [
                            {
                                "change_type": "unknown_type",
                                "target": "x",
                                "to_value": 1,
                                "reason": "test",
                            },
                            {
                                "change_type": "tighten_filter",
                                "target": "entry.conditions[indicator=rsi_14].value",
                                "to_value": 25,
                                "reason": "valid",
                            },
                        ],
                        "priority_score": 0.8,
                        "rationale": "test",
                    }
                ],
            },
        )
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        result = analyzer.reflect(_make_strategy(), _make_evaluation())
        assert len(result.proposed_changes) == 1
        assert result.proposed_changes[0].change_type == ChangeType.TIGHTEN_FILTER

    def test_replace_condition_is_normalized_into_supported_changes(self, llm):
        _mock_two_step_response(
            llm,
            {"root_causes": [], "diagnosis_summary": "test", "structural_issues": []},
            {
                "candidates": [
                    {
                        "hypothesis": {
                            "problem": "stale condition",
                            "hypothesis": "swap to a better momentum trigger",
                            "expected_outcome": "",
                            "confidence": 0.5,
                        },
                        "proposed_changes": [
                            {
                                "change_type": "replace_condition",
                                "target": "entry.conditions[indicator=rsi_14]",
                                "from_value": {"indicator": "rsi_14", "op": "<", "value": 30},
                                "to_value": {"indicator": "momentum_10d", "op": ">", "value": 0},
                                "reason": "replace stale trigger",
                            }
                        ],
                        "priority_score": 0.8,
                        "rationale": "test",
                    }
                ],
            },
        )
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert len(result.proposed_changes) == 2
        assert result.proposed_changes[0].change_type == ChangeType.REMOVE_CONDITION
        assert result.proposed_changes[1].change_type == ChangeType.ADD_CONDITION
        assert result.proposed_changes[1].to_value["indicator"] == "momentum_10d"

    def test_reflect_with_history_and_experience(self, llm):
        """History and experience context are injected into the LLM prompt."""
        captured_messages = []

        class CapturingLitellm:
            @staticmethod
            def completion(**kwargs):
                captured_messages.extend(kwargs.get("messages", []))
                import json

                class FakeMsg:
                    content = json.dumps(
                        {
                            "failure_patterns": [],
                            "reflection_summary": "ok",
                            "proposed_changes": [],
                        }
                    )

                class FakeChoice:
                    message = FakeMsg()

                class FakeResp:
                    choices = [FakeChoice()]

                return FakeResp()

        llm._litellm = CapturingLitellm()
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        history = "Round 1: score=30.0% — no improvement\n  Applied: tighten_filter on rsi"
        experience = "- [IMPROVED] test_v1 round 1: tighten_filter on rsi (30 → 25)"

        analyzer.reflect(
            _make_strategy(),
            _make_evaluation(),
            history_text=history,
            experience_text=experience,
        )

        # Both sections should appear in the user message
        user_content = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "Evolution History" in user_content
        assert "Round 1: score=30.0%" in user_content
        assert "Past Lessons" in user_content
        assert "IMPROVED" in user_content
        assert "Strategy Hypothesis Lens" in user_content
        assert "Current Assessment" in user_content

    def test_reflect_empty_history_no_section_headers(self, llm):
        """Empty history/experience → no section headers in prompt."""
        captured_messages = []

        class CapturingLitellm:
            @staticmethod
            def completion(**kwargs):
                captured_messages.extend(kwargs.get("messages", []))
                import json

                class FakeMsg:
                    content = json.dumps(
                        {
                            "failure_patterns": [],
                            "reflection_summary": "ok",
                            "proposed_changes": [],
                        }
                    )

                class FakeChoice:
                    message = FakeMsg()

                class FakeResp:
                    choices = [FakeChoice()]

                return FakeResp()

        llm._litellm = CapturingLitellm()
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        analyzer.reflect(
            _make_strategy(),
            _make_evaluation(),
            history_text="",
            experience_text="",
        )

        user_content = next(m["content"] for m in captured_messages if m["role"] == "user")
        assert "Evolution History" not in user_content
        assert "Past Lessons" not in user_content

    def test_two_step_reflection_uses_capped_timeout_budget(self):
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.timeout = 120
        llm.reflect_json.side_effect = [
            {
                "root_causes": [],
                "diagnosis_summary": "ok",
                "structural_issues": [],
            },
            {"candidates": []},
        ]
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        analyzer.reflect(_make_strategy(), _make_evaluation())

        first_call = llm.reflect_json.call_args_list[0]
        second_call = llm.reflect_json.call_args_list[1]
        assert first_call.kwargs["timeout"] == 45
        assert second_call.kwargs["timeout"] == 45
        assert first_call.kwargs["max_retries"] == 0
        assert second_call.kwargs["max_retries"] == 0

    def test_two_step_reflection_records_llm_telemetry(self):
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.timeout = 120
        llm.reflect_model = "test-reflect-model"
        llm.reflect_json.side_effect = [
            {
                "root_causes": [],
                "diagnosis_summary": "ok",
                "structural_issues": [],
            },
            {"candidates": []},
        ]
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert result.llm_telemetry is not None
        assert result.llm_telemetry.path == "two_step"
        assert len(result.llm_telemetry.calls) == 2
        assert result.llm_telemetry.calls[0].stage == "diagnosis"
        assert result.llm_telemetry.calls[1].stage == "experiment_design"
        assert result.llm_telemetry.calls[0].model == "test-reflect-model"
        assert result.llm_telemetry.total_duration_ms >= 0


# ── Heuristic fallback ───────────────────────────────────────────────


class TestHeuristicReflection:
    def test_low_win_rate_tightens_stop(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.35)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        assert any("win rate" in p.lower() for p in result.failure_patterns)
        sl_changes = [c for c in result.proposed_changes if "stop_loss" in c.target]
        assert len(sl_changes) >= 1

    def test_poor_pl_ratio_increases_target(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        evaluation = _make_evaluation(pl_ratio=1.0, avg_return=0.005)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        assert any("profit/loss" in p.lower() for p in result.failure_patterns)
        tp_changes = [c for c in result.proposed_changes if "take_profit" in c.target]
        assert len(tp_changes) >= 1

    def test_high_drawdown_flagged(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        # Use good P/L and decent win rate so drawdown branch triggers
        evaluation = _make_evaluation(
            max_drawdown=0.25,
            win_rate=0.50,
            pl_ratio=2.0,
            avg_return=0.03,
        )

        result = analyzer._reflect_heuristic(strategy, evaluation)

        assert any("drawdown" in p.lower() for p in result.failure_patterns)

    def test_few_signals_flagged(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        evaluation = _make_evaluation(signal_count=15)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        assert any("few signals" in p.lower() for p in result.failure_patterns)

    def test_few_signals_can_loosen_ma_period_tunable(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ]
        strategy.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=close_above_ma20].indicator",
                    range=[10, 60],
                    step=5,
                ),
            ]
        )
        evaluation = _make_evaluation(signal_count=15, win_rate=0.6)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        ma_changes = [c for c in result.proposed_changes if c.target.endswith(".indicator")]
        assert len(ma_changes) >= 1
        assert ma_changes[0].change_type == ChangeType.LOOSEN_FILTER
        assert ma_changes[0].to_value > ma_changes[0].from_value

    def test_few_signals_can_loosen_window_indicator_period_tunable(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.2),
        ]
        strategy.params = StrategyParams(
            tunable=[
                TunableParam(
                    target="entry.conditions[indicator=volume_ratio_1d_5d].indicator",
                    range=[3, 20],
                    step=1,
                ),
            ]
        )
        evaluation = _make_evaluation(signal_count=15, win_rate=0.6)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        window_changes = [c for c in result.proposed_changes if c.target.endswith(".indicator")]
        assert len(window_changes) >= 1
        assert window_changes[0].to_value > window_changes[0].from_value

    def test_few_signals_can_loosen_dual_ma_fast_and_slow(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        strategy.entry.conditions = [
            StrategyCondition(indicator="ma5_ge_ma10_or_crossing", op="==", value=True),
        ]
        strategy.params = StrategyParams(
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
        evaluation = _make_evaluation(signal_count=15, win_rate=0.5)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        fast_change = next(
            c for c in result.proposed_changes if c.target.endswith(".indicator.fast")
        )
        slow_change = next(
            c for c in result.proposed_changes if c.target.endswith(".indicator.slow")
        )
        assert fast_change.to_value < fast_change.from_value
        assert slow_change.to_value > slow_change.from_value

    def test_good_metrics_fine_tunes(self):
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        strategy = _make_strategy()
        evaluation = _make_evaluation(
            win_rate=0.60,
            pl_ratio=2.0,
            avg_return=0.03,
            max_drawdown=0.08,
            signal_count=100,
        )

        result = analyzer._reflect_heuristic(strategy, evaluation)

        assert any(
            "baseline" in p.lower() or "no obvious" in p.lower() for p in result.failure_patterns
        )

    def test_llm_failure_falls_back_to_heuristic(self, llm, monkeypatch):
        """When LLM fails, should fall back to heuristic."""

        def failing_reflect(*args, **kwargs):
            raise RuntimeError("LLM API error")

        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        monkeypatch.setattr(analyzer, "_reflect_llm", failing_reflect)

        strategy = _make_strategy()
        evaluation = _make_evaluation(win_rate=0.30)

        result = analyzer.reflect(strategy, evaluation)
        assert len(result.failure_patterns) > 0


# ── Failure formatting ───────────────────────────────────────────────


class TestFailureFormatting:
    def test_format_with_failures(self):
        ev = _make_evaluation()
        text = ReflectionAnalyzer._format_failures(ev)
        assert "AAPL" in text
        assert "stop_loss" in text

    def test_format_no_failures(self):
        ev = _make_evaluation()
        ev.failure_cases = []
        text = ReflectionAnalyzer._format_failures(ev)
        assert "No failure" in text


# ── Intensity and overfit awareness ──────────────────────────────────


class TestIntensityAndOverfit:
    """Test adaptive intensity and anti-overfit prompt injection."""

    def test_heuristic_intensity_scales_step(self):
        """Higher intensity should produce larger step changes."""
        from alphaevo.models.strategy import TunableParam

        strategy = _make_strategy()
        strategy.params.tunable = [
            TunableParam(
                target="entry.conditions[indicator=volume_ratio_1d_5d].value",
                range=[1.0, 3.0],
                step=0.1,
            ),
        ]
        evaluation = _make_evaluation(signal_count=10, win_rate=0.5)

        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)

        # Low intensity → small changes
        result_low = analyzer._reflect_heuristic(
            strategy,
            evaluation,
            intensity=0.5,
        )
        # High intensity → larger changes
        result_high = analyzer._reflect_heuristic(
            strategy,
            evaluation,
            intensity=2.0,
        )

        if result_low.proposed_changes and result_high.proposed_changes:
            low_delta = abs(
                result_low.proposed_changes[0].to_value - result_low.proposed_changes[0].from_value
            )
            high_delta = abs(
                result_high.proposed_changes[0].to_value
                - result_high.proposed_changes[0].from_value
            )
            assert high_delta >= low_delta

    def test_overfit_section_injected_when_data_present(self):
        """When anti_overfit has real data, overfit section should appear in prompt."""
        from unittest.mock import MagicMock

        from alphaevo.models.execution import AntiFitMetrics

        llm = MagicMock()
        # Two-step: first call = diagnosis, second call = experiments
        llm.reflect_json.side_effect = [
            {
                "root_causes": [{"problem": "overfit", "evidence": "gap", "severity": "high"}],
                "diagnosis_summary": "Overfitting detected",
                "structural_issues": [],
            },
            {"candidates": []},
        ]

        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        ev = _make_evaluation()
        ev.anti_overfit = AntiFitMetrics(
            train_win_rate=0.7,
            val_win_rate=0.4,
            test_win_rate=0.35,
            train_val_gap=0.30,
            val_test_gap=0.05,
        )

        analyzer.reflect(_make_strategy(), ev)
        # Check the diagnosis prompt (first call) contains overfit section
        call_args = llm.reflect_json.call_args_list[0][0][0]
        user_msg = call_args[1]["content"]
        assert "Generalization Analysis" in user_msg
        assert "HIGH" in user_msg  # train_val_gap > 0.10

    def test_overfit_section_absent_when_no_data(self):
        """When anti_overfit is default, no overfit section in prompt."""
        from unittest.mock import MagicMock

        llm = MagicMock()
        llm.reflect_json.return_value = {
            "failure_patterns": [],
            "reflection_summary": "OK",
            "proposed_changes": [],
        }

        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        ev = _make_evaluation()

        analyzer.reflect(_make_strategy(), ev)
        call_args = llm.reflect_json.call_args[0][0]
        user_msg = call_args[1]["content"]
        assert "Generalization Analysis" not in user_msg

    def test_format_failures_includes_indicator_snapshot(self):
        """_format_failures should include indicator values when present."""
        ev = _make_evaluation()
        ev.failure_cases = [
            TradeSignal(
                symbol="SNAP",
                signal_date=date_type(2025, 2, 1),
                direction="long",
                entry_price=100.0,
                exit_price=95.0,
                exit_reason="stop_loss",
                return_pct=-0.05,
                indicator_snapshot={"rsi_14": 65.1234, "ma5_above_ma10": True},
            ),
        ]
        result = ReflectionAnalyzer._format_failures(ev)
        assert "rsi_14=65.1234" in result
        assert "ma5_above_ma10=True" in result

    def test_change_logic_triggers_with_3_conditions_and_under_30_signals(self):
        """CHANGE_LOGIC should fire when entry has 3+ AND conditions and <30 signals."""
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=5)
        strategy = _make_strategy()
        strategy.entry.logic = "and"
        strategy.entry.conditions = [
            StrategyCondition(indicator="rsi_14", op="<", value=30),
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ]
        evaluation = _make_evaluation(signal_count=20)

        result = analyzer._reflect_heuristic(strategy, evaluation)

        logic_changes = [
            c for c in result.proposed_changes if c.change_type == ChangeType.CHANGE_LOGIC
        ]
        assert len(logic_changes) == 1
        assert logic_changes[0].to_value == "or"
