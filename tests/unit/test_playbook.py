"""Tests for ResearchPlaybook — playbook store and auto-discovery."""

from alphaevo.models.enums import ChangeType
from alphaevo.reflection.experience import ExperienceRecord, ExperienceStore
from alphaevo.reflection.playbook import (
    PlaybookStep,
    PlaybookStore,
    ResearchPlaybook,
    _classify_failure_pattern,
)


class TestPlaybookStore:
    def setup_method(self):
        self.store = PlaybookStore(db_path=":memory:")

    def test_builtins_are_seeded(self):
        """Built-in playbooks should be auto-loaded on init."""
        all_pb = self.store.get_all()
        assert len(all_pb) >= 5  # 6 builtins
        ids = {pb.playbook_id for pb in all_pb}
        assert "builtin_low_signals" in ids
        assert "builtin_high_drawdown" in ids

    def test_save_and_get(self):
        pb = ResearchPlaybook(
            playbook_id="test_pb",
            name="Test Playbook",
            trigger="signal_count < 5",
            problem_category="low_signals",
            steps=[
                PlaybookStep(action="loosen", target="entry", detail="Widen thresholds"),
            ],
        )
        self.store.save(pb)
        retrieved = self.store.get("test_pb")
        assert retrieved is not None
        assert retrieved.name == "Test Playbook"
        assert len(retrieved.steps) == 1

    def test_match_by_category(self):
        results = self.store.match("low_signals")
        assert len(results) >= 1
        assert all(r.problem_category == "low_signals" for r in results)

    def test_record_outcome(self):
        self.store.record_outcome("builtin_low_signals", succeeded=True)
        pb = self.store.get("builtin_low_signals")
        assert pb is not None
        assert pb.times_applied >= 1
        assert pb.times_succeeded >= 1

    def test_format_for_prompt(self):
        text = self.store.format_for_prompt(["low_signals", "high_drawdown"])
        assert "Research Playbooks" in text
        assert "Step 1:" in text

    def test_format_empty_category(self):
        text = self.store.format_for_prompt(["nonexistent_category"])
        assert text == ""


class TestPlaybookDiscovery:
    def test_discover_from_experience(self):
        exp_store = ExperienceStore(db_path=":memory:")
        pb_store = PlaybookStore(db_path=":memory:")

        # Create 5 successful records with "low win rate" failure pattern
        for i in range(5):
            exp_store.record(
                ExperienceRecord(
                    strategy_family="test_family",
                    strategy_id=f"test_v{i}",
                    round_num=i + 1,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions",
                    reason="Tighten RSI filter to improve quality",
                    score_before=0.30,
                    score_after=0.40,
                    score_delta=0.10,
                    worked=True,
                    failure_patterns=["Low win rate detected at 35%"],
                )
            )

        discovered = pb_store.discover_from_experience(exp_store, min_occurrences=3)
        assert len(discovered) >= 1
        assert discovered[0].problem_category == "low_win_rate"

    def test_discover_idempotent_no_collision(self):
        """Calling discover_from_experience twice should not produce ID collisions."""
        exp_store = ExperienceStore(db_path=":memory:")
        pb_store = PlaybookStore(db_path=":memory:")

        for i in range(5):
            exp_store.record(
                ExperienceRecord(
                    strategy_family="test",
                    strategy_id=f"test_v{i}",
                    round_num=i + 1,
                    change_type=ChangeType.TIGHTEN_FILTER,
                    target="entry.conditions",
                    reason="Tighten filter",
                    score_before=0.30,
                    score_after=0.40,
                    score_delta=0.10,
                    worked=True,
                    failure_patterns=["Low win rate detected"],
                )
            )

        # First call discovers playbooks
        first = pb_store.discover_from_experience(exp_store, min_occurrences=3)
        assert len(first) >= 1
        first_ids = {pb.playbook_id for pb in first}
        # All IDs should use content hash, not sequential index
        for pid in first_ids:
            assert "auto_" in pid

        # Second call should not discover new playbooks (same data)
        second = pb_store.discover_from_experience(exp_store, min_occurrences=3)
        assert len(second) == 0  # already exists in store

    def test_discover_insufficient_data(self):
        exp_store = ExperienceStore(db_path=":memory:")
        pb_store = PlaybookStore(db_path=":memory:")

        # Only 1 record — not enough
        exp_store.record(
            ExperienceRecord(
                strategy_family="test",
                strategy_id="test_v1",
                round_num=1,
                change_type=ChangeType.TIGHTEN_FILTER,
                target="entry.conditions",
                worked=True,
                failure_patterns=["Low win rate"],
            )
        )
        discovered = pb_store.discover_from_experience(exp_store, min_occurrences=3)
        assert len(discovered) == 0


class TestClassifyFailurePattern:
    def test_low_signals(self):
        assert _classify_failure_pattern("Too few signals to evaluate") == "low_signals"
        assert _classify_failure_pattern("Low signal count") == "low_signals"

    def test_high_drawdown(self):
        assert _classify_failure_pattern("High max_drawdown of 25%") == "high_drawdown"

    def test_low_win_rate(self):
        assert _classify_failure_pattern("Low win rate: 35%") == "low_win_rate"

    def test_win_rate_takes_priority_over_signals(self):
        """'Low win rate — too many false signals' should match win_rate, not signals."""
        assert (
            _classify_failure_pattern("Low win rate — too many false signals")
            == "low_win_rate"
        )

    def test_overfit(self):
        assert _classify_failure_pattern("Large train-val gap detected") == "overfit"

    def test_low_pl_ratio(self):
        assert _classify_failure_pattern("Poor profit/loss ratio") == "low_pl_ratio"

    def test_unknown(self):
        assert _classify_failure_pattern("Something completely different") is None
