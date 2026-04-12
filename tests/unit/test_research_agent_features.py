"""Targeted tests for research-agent upgrade features.

Covers:
- ResearchHypothesis / CandidateExperiment models
- Two-step LLM reflection (diagnose + experiment design)
- Analyzer._parse_candidates edge cases
- SelfCritic.rank_candidates method
- Extended ExperienceRecord fields (hypothesis, action_type, regime, source)
- Research summary injection into reflection context
- Fallback chain: two-step → single-step → heuristic
"""

import json
from datetime import date as date_type

import pytest

from alphaevo.core.config import AppConfig, LLMConfig
from alphaevo.core.llm import LLMClient
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    CandidateExperiment,
    EvaluationReport,
    OverallMetrics,
    ReflectionResult,
    ResearchHypothesis,
    StrategyChange,
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
from alphaevo.reflection.critic import SelfCritic
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore

# ── Helpers ──────────────────────────────────────────────────────────


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


def _make_evaluation(
    win_rate: float = 0.45,
    signal_count: int = 50,
    confidence: float = 0.35,
) -> EvaluationReport:
    return EvaluationReport(
        evaluation_id="eval-001",
        strategy_id="test_v1",
        overall=OverallMetrics(
            win_rate=win_rate,
            avg_return=0.01,
            profit_loss_ratio=1.2,
            max_drawdown=0.10,
            sharpe_ratio=0.8,
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


@pytest.fixture
def llm():
    config = LLMConfig(model="test-model")
    return LLMClient(config)


def _mock_llm_responses(llm, responses: list[dict]):
    """Mock LLM to return a sequence of JSON responses."""
    call_idx = [0]

    class FakeLitellm:
        @staticmethod
        def completion(**kwargs):
            idx = call_idx[0]
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


def _diagnosis_response(
    root_causes: list[dict] | None = None,
    summary: str = "Test diagnosis",
) -> dict:
    return {
        "root_causes": root_causes
        or [{"problem": "Low win rate", "evidence": "42%", "severity": "high"}],
        "diagnosis_summary": summary,
        "structural_issues": [],
    }


def _experiment_response(candidates: list[dict] | None = None) -> dict:
    if candidates is None:
        candidates = [
            {
                "hypothesis": {
                    "problem": "RSI too loose",
                    "hypothesis": "Tightening RSI filters false signals",
                    "expected_outcome": "+5% win rate",
                    "confidence": 0.7,
                },
                "proposed_changes": [
                    {
                        "change_type": "tighten_filter",
                        "target": "entry.conditions[indicator=rsi_14].value",
                        "from_value": 30,
                        "to_value": 25,
                        "reason": "Tighter RSI",
                    }
                ],
                "priority_score": 0.8,
                "rationale": "Most promising",
            }
        ]
    return {"candidates": candidates}


# ── Model tests ──────────────────────────────────────────────────────


class TestResearchHypothesis:
    def test_construction(self):
        h = ResearchHypothesis(
            problem="Low win rate",
            hypothesis="RSI threshold too high",
            expected_outcome="+5% win rate",
            confidence=0.8,
        )
        assert h.problem == "Low win rate"
        assert h.confidence == 0.8

    def test_defaults(self):
        h = ResearchHypothesis(problem="test", hypothesis="test")
        assert h.expected_outcome == ""
        assert h.confidence == 0.5


class TestCandidateExperiment:
    def test_construction(self):
        h = ResearchHypothesis(problem="test", hypothesis="test")
        change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="test",
        )
        c = CandidateExperiment(
            hypothesis=h,
            proposed_changes=[change],
            priority_score=0.9,
            rationale="Most promising",
        )
        assert c.priority_score == 0.9
        assert len(c.proposed_changes) == 1

    def test_defaults(self):
        h = ResearchHypothesis(problem="p", hypothesis="h")
        c = CandidateExperiment(hypothesis=h)
        assert c.proposed_changes == []
        assert c.priority_score == 0.0
        assert c.rationale == ""

    def test_reflection_result_backward_compat(self):
        """ReflectionResult with no candidates should work (backward compat)."""
        r = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-001",
            proposed_changes=[],
        )
        assert r.candidates == []
        assert r.diagnosis == ""


# ── Two-step LLM reflection tests ───────────────────────────────────


class TestTwoStepReflection:
    def test_produces_candidates_and_proposed_changes(self, llm):
        """Two-step path should populate both candidates and proposed_changes."""
        _mock_llm_responses(llm, [_diagnosis_response(), _experiment_response()])
        analyzer = ReflectionAnalyzer(llm, max_changes=3, num_candidates=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert len(result.candidates) == 1
        assert len(result.proposed_changes) == 1
        assert result.diagnosis == "Test diagnosis"
        assert result.failure_patterns == ["Low win rate"]

    def test_multiple_candidates_sorted_by_priority(self, llm):
        """Multiple candidates should be returned sorted by priority."""
        candidates = [
            {
                "hypothesis": {
                    "problem": "p1",
                    "hypothesis": "h1",
                    "expected_outcome": "",
                    "confidence": 0.5,
                },
                "proposed_changes": [
                    {
                        "change_type": "tighten_filter",
                        "target": "entry.conditions[indicator=rsi_14].value",
                        "from_value": 30,
                        "to_value": 25,
                        "reason": "r1",
                    }
                ],
                "priority_score": 0.3,
                "rationale": "Low priority",
            },
            {
                "hypothesis": {
                    "problem": "p2",
                    "hypothesis": "h2",
                    "expected_outcome": "",
                    "confidence": 0.9,
                },
                "proposed_changes": [
                    {
                        "change_type": "adjust_exit",
                        "target": "exit.stop_loss.value",
                        "from_value": 0.05,
                        "to_value": 0.04,
                        "reason": "r2",
                    }
                ],
                "priority_score": 0.95,
                "rationale": "High priority",
            },
        ]
        _mock_llm_responses(llm, [_diagnosis_response(), _experiment_response(candidates)])
        analyzer = ReflectionAnalyzer(llm, max_changes=3, num_candidates=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert len(result.candidates) == 2
        # Top candidate (highest priority) feeds proposed_changes
        assert result.candidates[0].priority_score == 0.95
        assert result.proposed_changes[0].target == "exit.stop_loss.value"

    def test_diagnosis_root_causes_become_failure_patterns(self, llm):
        """Root causes from diagnosis should map to failure_patterns."""
        causes = [
            {"problem": "Too few signals", "evidence": "12 < 30", "severity": "high"},
            {"problem": "Oversized drawdown", "evidence": "25%", "severity": "medium"},
        ]
        _mock_llm_responses(
            llm,
            [
                _diagnosis_response(root_causes=causes, summary="dual issues"),
                _experiment_response(),
            ],
        )
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert "Too few signals" in result.failure_patterns
        assert "Oversized drawdown" in result.failure_patterns

    def test_structural_issues_added_to_failure_patterns(self, llm):
        diag = {
            "root_causes": [],
            "diagnosis_summary": "structural problem",
            "structural_issues": ["Strategy logic inverted"],
        }
        _mock_llm_responses(llm, [diag, _experiment_response()])
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert "Strategy logic inverted" in result.failure_patterns

    def test_research_summary_injected_into_prompt(self, llm):
        """Research summary should appear in the LLM prompt context."""
        captured_messages = []

        class CapturingLitellm:
            @staticmethod
            def completion(**kwargs):
                captured_messages.extend(kwargs.get("messages", []))
                # Return valid diagnosis on first call, experiment on second
                idx = len(captured_messages) // 2  # rough counter
                if idx <= 1:
                    data = _diagnosis_response()
                else:
                    data = _experiment_response()

                class FakeMsg:
                    content = json.dumps(data)

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
            research_summary="Round 1 showed RSI=25 improved win_rate by 3%",
        )

        # The diagnosis prompt (first user message) should contain research summary
        user_messages = [m["content"] for m in captured_messages if m["role"] == "user"]
        assert any("Research Log" in msg for msg in user_messages)
        assert any("RSI=25 improved" in msg for msg in user_messages)


# ── Fallback chain tests ────────────────────────────────────────────


class TestReflectionFallbackChain:
    def test_falls_back_to_single_step_on_two_step_failure(self, llm):
        """If two-step fails (e.g. bad JSON), should fall back to single-step."""
        call_count = [0]

        class FailThenSucceedLitellm:
            @staticmethod
            def completion(**kwargs):
                call_count[0] += 1
                if call_count[0] <= 1:
                    # First call (two-step diagnosis) fails
                    raise RuntimeError("LLM error")
                # Third call (single-step fallback) succeeds
                data = {
                    "failure_patterns": ["fallback pattern"],
                    "reflection_summary": "fallback summary",
                    "proposed_changes": [
                        {
                            "change_type": "tighten_filter",
                            "target": "entry.conditions[indicator=rsi_14].value",
                            "from_value": 30,
                            "to_value": 25,
                            "reason": "from fallback",
                        }
                    ],
                }

                class FakeMsg:
                    content = json.dumps(data)

                class FakeChoice:
                    message = FakeMsg()

                class FakeResp:
                    choices = [FakeChoice()]

                return FakeResp()

        llm._litellm = FailThenSucceedLitellm()
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation())

        assert result.reflection_summary == "fallback summary"
        assert len(result.proposed_changes) == 1
        # No candidates from single-step fallback
        assert result.candidates == []
        assert result.llm_telemetry is not None
        assert result.llm_telemetry.path == "single_step_fallback"
        assert len(result.llm_telemetry.calls) == 2
        assert result.llm_telemetry.calls[0].success is False
        assert "LLM error" in result.llm_telemetry.fallback_trigger

    def test_falls_back_to_heuristic_when_all_llm_fails(self, llm):
        """If both LLM paths fail, heuristic should still produce changes."""

        class AlwaysFailLitellm:
            @staticmethod
            def completion(**kwargs):
                raise RuntimeError("LLM unavailable")

        llm._litellm = AlwaysFailLitellm()
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        result = analyzer.reflect(_make_strategy(), _make_evaluation(win_rate=0.30))

        assert len(result.proposed_changes) >= 1
        assert len(result.failure_patterns) >= 1
        assert result.llm_telemetry is not None
        assert result.llm_telemetry.path == "heuristic_fallback"
        assert len(result.llm_telemetry.calls) >= 2

    def test_research_summary_preserved_in_single_step_fallback(self, llm):
        """When two-step fails and single-step takes over, research_summary
        must still appear in the LLM prompt (not be silently dropped)."""
        captured_messages = []
        call_count = [0]

        class FailThenCaptureLitellm:
            @staticmethod
            def completion(**kwargs):
                call_count[0] += 1
                if call_count[0] <= 1:
                    raise RuntimeError("Two-step fails")
                captured_messages.extend(kwargs.get("messages", []))
                data = {
                    "failure_patterns": ["fallback pattern"],
                    "reflection_summary": "fallback summary",
                    "proposed_changes": [
                        {
                            "change_type": "tighten_filter",
                            "target": "entry.conditions[indicator=rsi_14].value",
                            "from_value": 30,
                            "to_value": 25,
                            "reason": "from fallback",
                        }
                    ],
                }

                class FakeMsg:
                    content = json.dumps(data)

                class FakeChoice:
                    message = FakeMsg()

                class FakeResp:
                    choices = [FakeChoice()]

                return FakeResp()

        llm._litellm = FailThenCaptureLitellm()
        analyzer = ReflectionAnalyzer(llm, max_changes=3)

        analyzer.reflect(
            _make_strategy(),
            _make_evaluation(),
            research_summary="Lesson from round 1: RSI oversold at 25 worked better",
        )

        user_msgs = [m["content"] for m in captured_messages if m["role"] == "user"]
        assert any("Research Log" in msg for msg in user_msgs), (
            "research_summary should appear as Research Log in fallback prompt"
        )
        assert any("RSI oversold at 25" in msg for msg in user_msgs), (
            "research_summary content should be included in fallback prompt"
        )


# ── _parse_candidates edge cases ─────────────────────────────────────


class TestParseCandidates:
    def test_empty_candidates_list(self, llm):
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        result = analyzer._parse_candidates([])
        assert result == []

    def test_candidate_with_missing_hypothesis_fields(self, llm):
        """Missing hypothesis fields should use defaults."""
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        raw = [
            {
                "hypothesis": {"problem": "test"},
                "proposed_changes": [],
                "priority_score": 0.5,
            }
        ]
        result = analyzer._parse_candidates(raw)
        assert len(result) == 1
        assert result[0].hypothesis.problem == "test"
        assert result[0].hypothesis.hypothesis == ""
        assert result[0].hypothesis.confidence == 0.5

    def test_candidate_with_invalid_change_types_filtered(self, llm):
        """Invalid change_type should be skipped within a candidate."""
        analyzer = ReflectionAnalyzer(llm, max_changes=3)
        raw = [
            {
                "hypothesis": {"problem": "p", "hypothesis": "h"},
                "proposed_changes": [
                    {"change_type": "bogus_type", "target": "x", "to_value": 1, "reason": "bad"},
                    {
                        "change_type": "tighten_filter",
                        "target": "entry.conditions[indicator=rsi_14].value",
                        "to_value": 25,
                        "reason": "good",
                    },
                ],
                "priority_score": 0.7,
            }
        ]
        result = analyzer._parse_candidates(raw)
        assert len(result) == 1
        assert len(result[0].proposed_changes) == 1

    def test_max_changes_enforced_per_candidate(self, llm):
        """Each candidate's changes should be capped at max_changes."""
        analyzer = ReflectionAnalyzer(llm, max_changes=2)
        many_changes = [
            {
                "change_type": "tighten_filter",
                "target": f"entry.conditions[indicator=ind_{i}].value",
                "to_value": i,
                "reason": f"change {i}",
            }
            for i in range(5)
        ]
        raw = [
            {
                "hypothesis": {"problem": "p", "hypothesis": "h"},
                "proposed_changes": many_changes,
                "priority_score": 0.8,
            }
        ]
        result = analyzer._parse_candidates(raw)
        assert len(result[0].proposed_changes) == 2


# ── SelfCritic.rank_candidates tests ────────────────────────────────


class TestRankCandidates:
    def _make_candidate(
        self,
        changes: list[StrategyChange],
        priority: float = 0.5,
        confidence: float = 0.6,
    ) -> CandidateExperiment:
        return CandidateExperiment(
            hypothesis=ResearchHypothesis(
                problem="test",
                hypothesis="test",
                confidence=confidence,
            ),
            proposed_changes=changes,
            priority_score=priority,
            rationale="test",
        )

    def test_empty_candidates(self):
        critic = SelfCritic()
        result = critic.rank_candidates(_make_strategy(), _make_evaluation(), [])
        assert result == []

    def test_candidates_sorted_by_score(self):
        critic = SelfCritic()
        low = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="low priority",
                )
            ],
            priority=0.2,
            confidence=0.3,
        )
        high = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.value",
                    from_value=0.05,
                    to_value=0.04,
                    reason="high priority",
                )
            ],
            priority=0.9,
            confidence=0.9,
        )
        result = critic.rank_candidates(_make_strategy(), _make_evaluation(), [low, high])
        assert len(result) == 2
        assert result[0].priority_score > result[1].priority_score

    def test_candidate_with_all_invalid_changes_removed(self):
        """A candidate whose changes are ALL rejected should be dropped."""
        critic = SelfCritic()
        bad = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target="entry.conditions",
                    to_value={"indicator": "totally_fake_indicator_xyz", "op": ">", "value": 1},
                    reason="unknown indicator",
                )
            ],
            priority=0.9,
        )
        good = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="valid change",
                )
            ],
            priority=0.5,
        )
        result = critic.rank_candidates(_make_strategy(), _make_evaluation(), [bad, good])
        # Only `good` should survive
        assert len(result) == 1
        assert result[0].proposed_changes[0].to_value == 25

    def test_novelty_penalty_with_experience_store(self):
        """Repeated failures should lower candidate score."""
        store = ExperienceStore(":memory:")
        # Record the same change as failed twice
        for round_num in (1, 2):
            store.record(
                ExperienceRecord(
                    strategy_family="test",
                    strategy_id="test_v1",
                    round_num=round_num,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="test",
                    worked=False,
                )
            )

        critic = SelfCritic(experience_store=store)

        repeated = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="same as before",
                )
            ],
            priority=0.8,
        )
        fresh = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.ADJUST_EXIT,
                    target="exit.stop_loss.value",
                    from_value=0.05,
                    to_value=0.04,
                    reason="new idea",
                )
            ],
            priority=0.8,
        )
        result = critic.rank_candidates(_make_strategy(), _make_evaluation(), [repeated, fresh])
        # `fresh` should score higher (no novelty penalty)
        assert result[0].proposed_changes[0].target == "exit.stop_loss.value"

    def test_multi_change_candidate_with_invalid_step_is_dropped(self):
        """Bundled multi-step candidates should not be partially applied."""
        critic = SelfCritic()
        mixed = self._make_candidate(
            [
                StrategyChange(
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions[indicator=rsi_14].value",
                    from_value=30,
                    to_value=25,
                    reason="valid",
                ),
                StrategyChange(
                    change_type=ChangeType.ADD_CONDITION,
                    target="entry.conditions",
                    to_value={"indicator": "nonexistent_xxx", "op": ">", "value": 1},
                    reason="invalid indicator",
                ),
            ],
            priority=0.7,
        )
        result = critic.rank_candidates(_make_strategy(), _make_evaluation(), [mixed])
        assert result == []


# ── Extended ExperienceRecord tests ──────────────────────────────────


class TestExtendedExperienceRecord:
    def test_new_fields_persist_and_load(self):
        """hypothesis, action_type, regime, source should roundtrip via SQLite."""
        store = ExperienceStore(":memory:")
        store.record(
            ExperienceRecord(
                strategy_family="test",
                strategy_id="test_v1",
                round_num=1,
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="test",
                score_before=0.30,
                score_after=0.40,
                score_delta=0.10,
                worked=True,
                hypothesis="RSI too loose lets in bad signals",
                action_type="llm",
                regime="trending_up",
                source="two_step_reflection",
            )
        )
        from alphaevo.reflection.experience import ExperienceQuery

        records = store.query(ExperienceQuery())
        assert len(records) == 1
        r = records[0]
        assert r.hypothesis == "RSI too loose lets in bad signals"
        assert r.action_type == "llm"
        assert r.regime == "trending_up"
        assert r.source == "two_step_reflection"

    def test_new_fields_default_empty(self):
        """Records without new fields should have empty defaults."""
        store = ExperienceStore(":memory:")
        store.record(
            ExperienceRecord(
                strategy_family="test",
                strategy_id="test_v1",
                round_num=1,
                change_type=ChangeType.ADJUST_EXIT,
                target="exit.stop_loss.value",
                worked=False,
            )
        )
        from alphaevo.reflection.experience import ExperienceQuery

        records = store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].hypothesis == ""
        assert records[0].action_type == ""
        assert records[0].regime == ""
        assert records[0].source == ""

    def test_batch_record_with_new_fields(self):
        """record_batch should also persist the new fields."""
        store = ExperienceStore(":memory:")
        records = [
            ExperienceRecord(
                strategy_family="test",
                strategy_id="test_v1",
                round_num=i,
                change_type=ChangeType.TIGHTEN_FILTER,
                target="t",
                hypothesis=f"hyp_{i}",
                action_type="param_search",
                regime="trending_down",
                source="pipeline",
                worked=i % 2 == 0,
            )
            for i in range(3)
        ]
        store.record_batch(records)

        from alphaevo.reflection.experience import ExperienceQuery

        loaded = store.query(ExperienceQuery())
        assert len(loaded) == 3
        # Query returns newest first (ORDER BY created_at DESC)
        hypotheses = {r.hypothesis for r in loaded}
        assert hypotheses == {"hyp_0", "hyp_1", "hyp_2"}
        assert all(r.action_type == "param_search" for r in loaded)
        assert all(r.regime == "trending_down" for r in loaded)

    def test_migration_adds_columns(self):
        """init_db should migrate old tables missing new columns."""
        import sqlite3

        # Create a DB with the old schema (no new columns)
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS evolution_experience (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_family TEXT NOT NULL,
                strategy_id TEXT NOT NULL,
                round_num INTEGER NOT NULL,
                change_type TEXT NOT NULL,
                target TEXT NOT NULL,
                from_value TEXT,
                to_value TEXT,
                reason TEXT DEFAULT '',
                score_before REAL DEFAULT 0.0,
                score_after REAL DEFAULT 0.0,
                score_delta REAL DEFAULT 0.0,
                worked INTEGER DEFAULT 0,
                failure_patterns TEXT DEFAULT '[]',
                lesson TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """INSERT INTO evolution_experience
            (strategy_family, strategy_id, round_num, change_type, target, created_at)
            VALUES ('fam', 'id', 1, 'tighten_filter', 't', '2024-01-01')"""
        )
        conn.commit()

        # Now create ExperienceStore using the same DB
        store = ExperienceStore.__new__(ExperienceStore)
        store._is_memory = True
        store._db_path = ":memory:"
        store._shared_conn = conn
        store.init_db()

        # Verify the new columns exist
        cols = {
            row[1] for row in conn.execute("PRAGMA table_info(evolution_experience)").fetchall()
        }
        assert "hypothesis" in cols
        assert "action_type" in cols
        assert "regime" in cols
        assert "source" in cols


# ── Evolution pipeline pending experience with new fields ────────────


class TestPendingExperienceNewFields:
    def test_hypothesis_and_source_recorded(self):
        """_record_pending_experience should persist hypothesis, source, regime."""
        config = AppConfig()
        from alphaevo.orchestrator.evolution import EvolutionPipeline

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="test",
        )
        pipeline._record_pending_experience(
            [
                (
                    change,
                    0.30,
                    "test_v2",
                    1,
                    "RSI too loose causes false signals",
                    "two_step_llm",
                    "trending_up",
                )
            ],
            family_id="test",
            score_after=0.45,
            worked=True,
        )

        from alphaevo.reflection.experience import ExperienceQuery

        records = pipeline._experience_store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].hypothesis == "RSI too loose causes false signals"
        assert records[0].action_type == "two_step_llm"
        assert records[0].source == "two_step_llm"
        assert records[0].regime == "trending_up"

    def test_empty_new_fields_still_works(self):
        """Empty strings for new fields should be fine."""
        config = AppConfig()
        from alphaevo.orchestrator.evolution import EvolutionPipeline

        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        change = StrategyChange(
            change_type=ChangeType.ADJUST_EXIT,
            target="exit.stop_loss.value",
            from_value=0.05,
            to_value=0.04,
            reason="test",
        )
        pipeline._record_pending_experience(
            [(change, 0.35, "test_v2", 1, "", "", "")],
            family_id="test",
            score_after=0.40,
            worked=True,
        )

        from alphaevo.reflection.experience import ExperienceQuery

        records = pipeline._experience_store.query(ExperienceQuery())
        assert len(records) == 1
        assert records[0].hypothesis == ""
        assert records[0].source == ""


# ── heuristic_only classmethod ──────────────────────────────────────


class TestHeuristicOnly:
    def test_heuristic_only_creates_valid_analyzer(self):
        """heuristic_only() should create an analyzer with all attributes set."""
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=5)
        assert analyzer.max_changes == 5
        assert analyzer.llm is None
        assert analyzer.num_candidates == 0
        assert analyzer._serializer is not None

    def test_heuristic_only_reflect_works(self):
        """heuristic_only analyzer should produce changes via heuristic path."""
        analyzer = ReflectionAnalyzer.heuristic_only(max_changes=3)
        result = analyzer.reflect(_make_strategy(), _make_evaluation(win_rate=0.30))
        # Should use heuristic fallback (llm is None)
        assert len(result.proposed_changes) >= 1
        assert len(result.failure_patterns) >= 1


# ── Enriched format_for_prompt ──────────────────────────────────────


class TestEnrichedFormatForPrompt:
    def test_hypothesis_in_prompt_output(self):
        """format_for_prompt should include hypothesis when available."""
        store = ExperienceStore(":memory:")
        store.record(
            ExperienceRecord(
                strategy_family="fam",
                strategy_id="fam_v1",
                round_num=1,
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions[indicator=rsi_14].value",
                from_value=30,
                to_value=25,
                reason="test",
                score_before=0.30,
                score_after=0.40,
                score_delta=0.10,
                worked=True,
                hypothesis="RSI too loose lets in bad signals",
                action_type="llm",
                regime="trending_up",
                source="two_step_reflection",
            )
        )
        text = store.format_for_prompt(family_id="fam", limit=10)
        assert "Hypothesis: RSI too loose" in text
        assert "Regime: trending_up" in text
        assert "Method: llm" in text

    def test_empty_new_fields_no_extra_lines(self):
        """Records without hypothesis/regime should not show empty labels."""
        store = ExperienceStore(":memory:")
        store.record(
            ExperienceRecord(
                strategy_family="fam",
                strategy_id="fam_v1",
                round_num=1,
                change_type=ChangeType.TIGHTEN_FILTER,
                target="t",
                worked=True,
            )
        )
        text = store.format_for_prompt(family_id="fam", limit=10)
        assert "Hypothesis:" not in text
        assert "Regime:" not in text
        assert "Method:" not in text


# ── Candidate fallback on mutation failure ──────────────────────────


class TestCandidateFallbackMutation:
    def test_fallback_to_second_candidate_on_mutation_error(self):
        """When top candidate fails mutation, pipeline should try next."""
        from unittest.mock import MagicMock

        from alphaevo.orchestrator.evolution import EvolutionPipeline
        from alphaevo.reflection.mutator import MutationError

        config = AppConfig()
        pipeline = EvolutionPipeline(config)
        pipeline._experience_store = ExperienceStore(":memory:")

        strategy = _make_strategy()
        evaluation = _make_evaluation()

        # Build a reflection with 2 candidates
        cand1_change = StrategyChange(
            change_type=ChangeType.ADD_CONDITION,
            target="entry.conditions",
            to_value={"indicator": "nonexistent_xyz", "op": ">", "value": 1},
            reason="will fail mutation",
        )
        cand2_change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[indicator=rsi_14].value",
            from_value=30,
            to_value=25,
            reason="will succeed",
        )
        reflection = ReflectionResult(
            strategy_id="test_v1",
            evaluation_id="eval-001",
            proposed_changes=cand1_change.model_copy() and [cand1_change],
            candidates=[
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p1", hypothesis="h1"),
                    proposed_changes=[cand1_change],
                    priority_score=0.9,
                ),
                CandidateExperiment(
                    hypothesis=ResearchHypothesis(problem="p2", hypothesis="h2"),
                    proposed_changes=[cand2_change],
                    priority_score=0.7,
                ),
            ],
        )

        call_count = [0]
        original_mutate = pipeline._mutator.mutate

        def mock_mutate(strat, changes, atomic=False):
            call_count[0] += 1
            if call_count[0] == 1:
                raise MutationError("Unknown indicator")
            return original_mutate(strat, changes, atomic=atomic)

        pipeline._mutator.mutate = mock_mutate
        pipeline.store.save = MagicMock()

        # Simulate the mutation step from evolve_async
        from datetime import date

        from alphaevo.models.execution import BacktestResult, SampleBatch
        from alphaevo.orchestrator.pipeline import RunResult

        run_result = RunResult(
            strategy=strategy,
            batch=SampleBatch(
                batch_id="b",
                strategy_id="test_v1",
                symbols=["AAPL"],
                date_range=(date(2024, 1, 1), date(2024, 12, 31)),
            ),
            backtest_result=BacktestResult(strategy_id="test_v1", batch_id="b", signals=[]),
            evaluation=evaluation,
        )

        # Run the candidate fallback logic directly
        mutated = False
        candidates_to_try = list(reflection.candidates)
        for _cand_idx, candidate in enumerate(candidates_to_try):
            changes_to_apply = candidate.proposed_changes
            if not changes_to_apply:
                continue
            try:
                new_strategy = pipeline._mutator.mutate(run_result.strategy, changes_to_apply)
                pipeline.store.save(new_strategy)
                reflection.proposed_changes = changes_to_apply
                mutated = True
                break
            except MutationError:
                continue

        assert mutated
        # Candidate #2 should have been used
        assert call_count[0] == 2
        assert reflection.proposed_changes == [cand2_change]
