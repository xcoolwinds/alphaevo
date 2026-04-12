"""Tests for StrategyStore (SQLite-backed strategy persistence)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphaevo.models.enums import MarketType, StrategyCategory
from alphaevo.models.execution import EvaluationReport, OverallMetrics
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)
from alphaevo.strategy.store import StrategyStore

BUILTIN_DIR = Path(__file__).parent.parent.parent / "strategies" / "builtin"


def _make_strategy(
    sid: str = "test_strat_v1",
    name: str = "Test Strategy",
    version: int = 1,
    parent_id: str | None = None,
    category: StrategyCategory = StrategyCategory.TREND,
) -> Strategy:
    """Helper to create a minimal valid Strategy."""
    return Strategy(
        meta=StrategyMeta(
            id=sid,
            name=name,
            version=version,
            parent_id=parent_id,
            market=MarketType.A_SHARE,
            category=category,
            tags=["test"],
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
        ),
        description="A test strategy.",
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
    )


def _make_eval_report(
    strategy_id: str = "test_strat_v1",
    evaluation_id: str = "eval_001",
    confidence: float = 0.75,
    win_rate: float = 0.6,
    signal_count: int = 50,
) -> EvaluationReport:
    """Helper to create a minimal EvaluationReport."""
    return EvaluationReport(
        evaluation_id=evaluation_id,
        strategy_id=strategy_id,
        batch_id="batch_001",
        overall=OverallMetrics(
            win_rate=win_rate,
            avg_return=0.03,
            profit_loss_ratio=1.8,
            max_drawdown=0.12,
            sharpe_ratio=1.5,
            signal_count=signal_count,
        ),
        confidence_score=confidence,
        created_at=datetime(2025, 6, 1, tzinfo=timezone.utc),
    )


class TestStrategyStore:
    """Tests for StrategyStore with in-memory SQLite."""

    def setup_method(self) -> None:
        self.store = StrategyStore(db_path=":memory:")

    # ── Save / Get round-trip ─────────────────────────────────────────

    def test_save_and_get_roundtrip(self) -> None:
        original = _make_strategy()
        self.store.save(original)

        restored = self.store.get(original.meta.id)
        assert restored is not None
        assert restored.meta.id == original.meta.id
        assert restored.meta.name == original.meta.name
        assert restored.meta.version == original.meta.version
        assert restored.meta.category == original.meta.category
        assert len(restored.entry.conditions) == len(original.entry.conditions)
        assert restored.exit.stop_loss.value == original.exit.stop_loss.value
        assert restored.description == original.description

    def test_get_nonexistent_returns_none(self) -> None:
        assert self.store.get("nonexistent_strategy") is None

    def test_save_upsert_updates_existing(self) -> None:
        s = _make_strategy()
        self.store.save(s)

        # Modify name and re-save
        s_updated = _make_strategy(name="Updated Name")
        self.store.save(s_updated)

        restored = self.store.get(s.meta.id)
        assert restored is not None
        assert restored.meta.name == "Updated Name"

        # Should still be only one row
        assert len(self.store.list_all()) == 1

    # ── List ──────────────────────────────────────────────────────────

    def test_list_all(self) -> None:
        self.store.save(_make_strategy("strat_a_v1", "Strategy A"))
        self.store.save(_make_strategy("strat_b_v1", "Strategy B"))
        self.store.save(_make_strategy("strat_c_v1", "Strategy C"))

        result = self.store.list_all()
        assert len(result) == 3
        ids = {s.meta.id for s in result}
        assert ids == {"strat_a_v1", "strat_b_v1", "strat_c_v1"}

    def test_list_all_empty(self) -> None:
        assert self.store.list_all() == []

    def test_list_by_family(self) -> None:
        self.store.save(_make_strategy("trend_v1", "Trend v1", version=1))
        self.store.save(_make_strategy("trend_v2", "Trend v2", version=2, parent_id="trend_v1"))
        self.store.save(_make_strategy("other_v1", "Other v1", version=1))

        family = self.store.list_by_family("trend")
        assert len(family) == 2
        assert family[0].meta.id == "trend_v1"
        assert family[1].meta.id == "trend_v2"

    def test_list_by_family_empty(self) -> None:
        assert self.store.list_by_family("nonexistent") == []

    # ── Delete ────────────────────────────────────────────────────────

    def test_delete_existing(self) -> None:
        self.store.save(_make_strategy())
        assert self.store.delete("test_strat_v1") is True
        assert self.store.get("test_strat_v1") is None

    def test_delete_nonexistent(self) -> None:
        assert self.store.delete("nonexistent") is False

    def test_delete_does_not_affect_others(self) -> None:
        self.store.save(_make_strategy("keep_me_v1", "Keep"))
        self.store.save(_make_strategy("delete_me_v1", "Delete"))

        self.store.delete("delete_me_v1")
        assert self.store.get("keep_me_v1") is not None
        assert len(self.store.list_all()) == 1

    # ── Evaluations ───────────────────────────────────────────────────

    def test_save_and_get_evaluations(self) -> None:
        self.store.save(_make_strategy())
        report = _make_eval_report()
        self.store.save_evaluation(report)

        evals = self.store.get_evaluations("test_strat_v1")
        assert len(evals) == 1
        assert evals[0].evaluation_id == "eval_001"
        assert evals[0].confidence_score == 0.75
        assert evals[0].overall.win_rate == 0.6
        assert evals[0].overall.signal_count == 50

    def test_get_evaluations_empty(self) -> None:
        assert self.store.get_evaluations("no_such_strategy") == []

    def test_multiple_evaluations_per_strategy(self) -> None:
        self.store.save(_make_strategy())
        self.store.save_evaluation(_make_eval_report(evaluation_id="eval_a", confidence=0.5))
        self.store.save_evaluation(_make_eval_report(evaluation_id="eval_b", confidence=0.8))

        evals = self.store.get_evaluations("test_strat_v1")
        assert len(evals) == 2

    # ── Leaderboard ───────────────────────────────────────────────────

    def test_get_leaderboard_ordering(self) -> None:
        # Create 3 strategies with different scores
        for label, score in [("alpha", 0.9), ("beta", 0.5), ("gamma", 0.7)]:
            sid = f"{label}_v1"
            self.store.save(_make_strategy(sid, label.title()))
            self.store.save_evaluation(
                _make_eval_report(strategy_id=sid, evaluation_id=f"eval_{label}", confidence=score)
            )

        board = self.store.get_leaderboard(limit=10)
        assert len(board) == 3
        scores = [report.confidence_score for _, report in board]
        assert scores == sorted(scores, reverse=True)
        assert board[0][0].meta.id == "alpha_v1"
        assert board[1][0].meta.id == "gamma_v1"
        assert board[2][0].meta.id == "beta_v1"

    def test_get_leaderboard_respects_limit(self) -> None:
        for i in range(5):
            sid = f"strat_{i}_v1"
            self.store.save(_make_strategy(sid, f"Strat {i}"))
            self.store.save_evaluation(
                _make_eval_report(strategy_id=sid, evaluation_id=f"eval_{i}", confidence=i * 0.1)
            )

        board = self.store.get_leaderboard(limit=2)
        assert len(board) == 2

    def test_get_leaderboard_empty(self) -> None:
        assert self.store.get_leaderboard() == []

    def test_get_leaderboard_filters_low_signal(self) -> None:
        self.store.save(_make_strategy("high_score_low_signal_v1", "Low Signal"))
        self.store.save_evaluation(
            _make_eval_report(
                strategy_id="high_score_low_signal_v1",
                evaluation_id="eval_low_signal",
                confidence=0.95,
                signal_count=8,
            )
        )

        self.store.save(_make_strategy("lower_score_valid_signal_v1", "Valid Signal"))
        self.store.save_evaluation(
            _make_eval_report(
                strategy_id="lower_score_valid_signal_v1",
                evaluation_id="eval_valid_signal",
                confidence=0.70,
                signal_count=45,
            )
        )

        board = self.store.get_leaderboard(limit=10, min_signal_count=30)
        assert len(board) == 1
        assert board[0][0].meta.id == "lower_score_valid_signal_v1"

    # ── Import ────────────────────────────────────────────────────────

    def test_import_from_file(self) -> None:
        if not BUILTIN_DIR.is_dir():
            pytest.skip("Builtin strategies directory not found")

        yaml_path = BUILTIN_DIR / "mean_reversion_oversold.yaml"
        if not yaml_path.exists():
            pytest.skip("mean_reversion_oversold.yaml not found")

        strategy = self.store.import_from_file(yaml_path)
        assert strategy.meta.id == "mean_reversion_oversold_v1"

        # Should be persisted
        restored = self.store.get("mean_reversion_oversold_v1")
        assert restored is not None
        assert restored.meta.name == strategy.meta.name

    def test_import_builtin_strategies(self) -> None:
        if not BUILTIN_DIR.is_dir():
            pytest.skip("Builtin strategies directory not found")

        count = self.store.import_builtin_strategies(BUILTIN_DIR)
        assert count >= 4

        all_strategies = self.store.list_all()
        assert len(all_strategies) >= 4

    def test_import_builtin_nonexistent_dir(self) -> None:
        count = self.store.import_builtin_strategies(Path("/nonexistent/dir"))
        assert count == 0

    # ── DB initialization ─────────────────────────────────────────────

    def test_init_creates_tables(self) -> None:
        """Tables should exist after init."""
        store = StrategyStore(db_path=":memory:")

        # Verify by saving — would fail if tables don't exist
        store.save(_make_strategy())
        assert store.get("test_strat_v1") is not None


class TestStrategyStoreEdgeCases:
    """Edge-case and integration tests."""

    def setup_method(self) -> None:
        self.store = StrategyStore(db_path=":memory:")

    def test_strategy_with_parent_id(self) -> None:
        parent = _make_strategy("base_v1", "Base", version=1)
        child = _make_strategy("base_v2", "Base v2", version=2, parent_id="base_v1")

        self.store.save(parent)
        self.store.save(child)

        restored_child = self.store.get("base_v2")
        assert restored_child is not None
        assert restored_child.meta.parent_id == "base_v1"
        assert restored_child.meta.version == 2

    def test_evaluation_upsert(self) -> None:
        self.store.save(_make_strategy())
        report = _make_eval_report(confidence=0.5)
        self.store.save_evaluation(report)

        # Update with higher confidence
        updated = _make_eval_report(confidence=0.9)
        self.store.save_evaluation(updated)

        evals = self.store.get_evaluations("test_strat_v1")
        assert len(evals) == 1
        assert evals[0].confidence_score == 0.9

    def test_different_categories(self) -> None:
        for cat in [StrategyCategory.TREND, StrategyCategory.REVERSAL]:
            sid = f"{cat.value}_v1"
            self.store.save(_make_strategy(sid, cat.value.title(), category=cat))

        all_strats = self.store.list_all()
        assert len(all_strats) == 2
