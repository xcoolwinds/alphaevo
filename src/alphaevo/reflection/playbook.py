"""Research Playbook — codified procedural strategies for recurring problems.

Inspired by Hermes-Agent's "programmatic skills": instead of just storing
_what worked_ (PatternLibrary) or _individual lessons_ (ExperienceStore),
Playbooks capture _how to diagnose and fix a class of problem_.

A Playbook is a reusable procedure:
  "When you see <trigger_condition>, try <steps> in order."

Playbooks are auto-discovered from ExperienceStore patterns and can also
be hand-authored for domain knowledge (e.g., "how to diagnose high drawdown").

This gives the LLM structured guidance instead of raw lesson dumps.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Models ──────────────────────────────────────────────────────────


class PlaybookStep(BaseModel):
    """A single step in a research playbook."""

    action: str  # e.g. "loosen_filter", "switch_logic", "add_indicator"
    target: str  # e.g. "entry.conditions", "entry.logic"
    detail: str  # human-readable explanation
    typical_value: Any = None  # representative value from successful cases


class ResearchPlaybook(BaseModel):
    """A reusable procedure for fixing a class of strategy problem.

    Example::

        ResearchPlaybook(
            playbook_id="fix_low_signal_count",
            name="How to fix low signal count",
            trigger="signal_count < 20 AND entry.logic == 'and'",
            problem_category="low_signals",
            steps=[
                PlaybookStep(action="switch_logic", target="entry.logic",
                             detail="Switch AND→OR to relax conjunction"),
                PlaybookStep(action="loosen_filter", target="entry.conditions",
                             detail="Widen the weakest threshold by 20%"),
            ],
            success_rate=0.65,
            times_applied=12,
        )
    """

    playbook_id: str
    name: str
    trigger: str  # Natural-language condition description
    problem_category: str  # e.g. "low_signals", "high_drawdown", "low_win_rate", "overfit"
    steps: list[PlaybookStep] = Field(default_factory=list)
    success_rate: float = 0.0
    times_applied: int = 0
    times_succeeded: int = 0
    source: str = "auto"  # "auto" | "builtin" | "user"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Built-in Playbooks (domain knowledge) ───────────────────────────

_BUILTIN_PLAYBOOKS: list[ResearchPlaybook] = [
    ResearchPlaybook(
        playbook_id="builtin_low_signals",
        name="Low signal count recovery",
        trigger="signal_count < 25 AND entry conditions >= 3",
        problem_category="low_signals",
        steps=[
            PlaybookStep(
                action="switch_logic",
                target="entry.logic",
                detail="If logic is AND with 3+ conditions, try switching to OR",
            ),
            PlaybookStep(
                action="loosen_filter",
                target="entry.conditions",
                detail="Widen the strictest threshold (highest rejection rate) by 15-25%",
            ),
            PlaybookStep(
                action="remove_condition",
                target="entry.conditions",
                detail="If still <15 signals, remove the condition contributing least to win rate",
            ),
        ],
        source="builtin",
    ),
    ResearchPlaybook(
        playbook_id="builtin_high_drawdown",
        name="High drawdown diagnosis",
        trigger="max_drawdown > 0.20",
        problem_category="high_drawdown",
        steps=[
            PlaybookStep(
                action="tighten_stop",
                target="exit.stop_loss.value",
                detail="First check: is stop loss too wide? Try reducing by 1-2 percentage points",
            ),
            PlaybookStep(
                action="add_filter",
                target="entry.conditions",
                detail="Add a volatility filter (e.g. volatility_20d < 0.04) to avoid turbulent entries",
            ),
            PlaybookStep(
                action="reduce_holding",
                target="exit.max_holding_days",
                detail="Reduce max holding period to lock in gains before reversal",
            ),
        ],
        source="builtin",
    ),
    ResearchPlaybook(
        playbook_id="builtin_low_win_rate",
        name="Low win rate improvement",
        trigger="win_rate < 0.45",
        problem_category="low_win_rate",
        steps=[
            PlaybookStep(
                action="tighten_filter",
                target="entry.conditions",
                detail="Add quality filters: RSI extremes, trend confirmation, volume spikes",
            ),
            PlaybookStep(
                action="add_indicator",
                target="entry.conditions",
                detail="Add a momentum or trend indicator (ma20_slope > 0, momentum_10d > 0)",
            ),
            PlaybookStep(
                action="adjust_exit",
                target="exit.take_profit",
                detail="If P/L ratio is OK but win rate is low, lower take-profit target",
            ),
        ],
        source="builtin",
    ),
    ResearchPlaybook(
        playbook_id="builtin_overfit",
        name="Overfit detection and mitigation",
        trigger="train_val_gap > 0.10 OR param_sensitivity > 0.30",
        problem_category="overfit",
        steps=[
            PlaybookStep(
                action="remove_condition",
                target="entry.conditions",
                detail="Reduce complexity: remove the most recently added condition",
            ),
            PlaybookStep(
                action="widen_threshold",
                target="entry.conditions",
                detail="Round thresholds to coarser values (e.g. 0.0347 → 0.035)",
            ),
            PlaybookStep(
                action="simplify_exit",
                target="exit",
                detail="Replace composite exit with simple pct stop-loss + rr take-profit",
            ),
        ],
        source="builtin",
    ),
    ResearchPlaybook(
        playbook_id="builtin_low_pl_ratio",
        name="Poor profit/loss ratio fix",
        trigger="profit_loss_ratio < 1.5 AND win_rate >= 0.45",
        problem_category="low_pl_ratio",
        steps=[
            PlaybookStep(
                action="widen_tp",
                target="exit.take_profit.value",
                detail="Increase take-profit target (raise R:R from current to +0.5)",
            ),
            PlaybookStep(
                action="add_trend_filter",
                target="entry.conditions",
                detail="Add trend alignment filter so entries are with the trend, not against",
            ),
            PlaybookStep(
                action="tighten_stop",
                target="exit.stop_loss.value",
                detail="Tighten stop loss to cut losers faster (reduce by 0.5-1 pct point)",
            ),
        ],
        source="builtin",
    ),
    ResearchPlaybook(
        playbook_id="builtin_proxy_strategy_diagnosis",
        name="Proxy/experimental strategy diagnosis",
        trigger="strategy uses proxy indicators (experimental=true)",
        problem_category="proxy_degradation",
        steps=[
            PlaybookStep(
                action="check_proxy_quality",
                target="entry.conditions",
                detail="Verify proxy indicators correlate with the real signal (check win_rate vs random baseline)",
            ),
            PlaybookStep(
                action="add_price_anchor",
                target="entry.conditions",
                detail="Supplement proxy with hard price/volume conditions as reality checks",
            ),
            PlaybookStep(
                action="lower_confidence",
                target="evaluation",
                detail="Apply extra confidence penalty for proxy-dependent results",
            ),
        ],
        source="builtin",
    ),
]


# ── SQLite Schema ───────────────────────────────────────────────────

_PLAYBOOK_SCHEMA = """
CREATE TABLE IF NOT EXISTS research_playbooks (
    playbook_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    trigger_condition TEXT NOT NULL,
    problem_category TEXT NOT NULL,
    steps TEXT NOT NULL DEFAULT '[]',
    success_rate REAL DEFAULT 0.0,
    times_applied INTEGER DEFAULT 0,
    times_succeeded INTEGER DEFAULT 0,
    source TEXT DEFAULT 'auto',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_playbook_category
    ON research_playbooks(problem_category);
CREATE INDEX IF NOT EXISTS idx_playbook_success
    ON research_playbooks(success_rate DESC);
"""


# ── PlaybookStore ───────────────────────────────────────────────────


class PlaybookStore:
    """SQLite-backed repository of research playbooks.

    Shares the same DB as ExperienceStore / PatternLibrary.
    """

    def __init__(self, db_path: str | Path = "~/.alphaevo/alphaevo.db") -> None:
        self._is_memory = str(db_path) == ":memory:"
        if self._is_memory:
            self._db_path = ":memory:"
            self._shared_conn: sqlite3.Connection | None = sqlite3.connect(":memory:")
        else:
            self._db_path = str(Path(db_path).expanduser().resolve())
            Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
            self._shared_conn = None
        self._init_db()

    # ── lifecycle ────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_PLAYBOOK_SCHEMA)
        # Seed builtins if missing
        for pb in _BUILTIN_PLAYBOOKS:
            if self.get(pb.playbook_id) is None:
                self.save(pb)

    # ── CRUD ─────────────────────────────────────────────────────────

    def save(self, playbook: ResearchPlaybook) -> None:
        """Insert or update a playbook."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO research_playbooks
                    (playbook_id, name, trigger_condition, problem_category,
                     steps, success_rate, times_applied, times_succeeded,
                     source, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    playbook.playbook_id,
                    playbook.name,
                    playbook.trigger,
                    playbook.problem_category,
                    json.dumps([s.model_dump() for s in playbook.steps]),
                    playbook.success_rate,
                    playbook.times_applied,
                    playbook.times_succeeded,
                    playbook.source,
                    playbook.created_at.isoformat(),
                    playbook.updated_at.isoformat(),
                ),
            )

    def get(self, playbook_id: str) -> ResearchPlaybook | None:
        """Retrieve a playbook by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM research_playbooks WHERE playbook_id = ?",
                (playbook_id,),
            ).fetchone()
        return self._row_to_playbook(row) if row else None

    def get_all(self) -> list[ResearchPlaybook]:
        """Return all playbooks ordered by success rate."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_playbooks ORDER BY success_rate DESC"
            ).fetchall()
        return [self._row_to_playbook(r) for r in rows]

    def match(
        self,
        problem_category: str,
        *,
        min_success_rate: float = 0.0,
        limit: int = 5,
    ) -> list[ResearchPlaybook]:
        """Find playbooks that address a specific problem category."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM research_playbooks "
                "WHERE problem_category = ? AND success_rate >= ? "
                "ORDER BY success_rate DESC LIMIT ?",
                (problem_category, min_success_rate, limit),
            ).fetchall()
        return [self._row_to_playbook(r) for r in rows]

    def record_outcome(self, playbook_id: str, succeeded: bool) -> None:
        """Track whether applying a playbook led to improvement."""
        with self._connect() as conn:
            if succeeded:
                conn.execute(
                    "UPDATE research_playbooks SET "
                    "times_applied = times_applied + 1, "
                    "times_succeeded = times_succeeded + 1, "
                    "success_rate = CAST(times_succeeded + 1 AS REAL) / (times_applied + 1), "
                    "updated_at = ? "
                    "WHERE playbook_id = ?",
                    (datetime.now(timezone.utc).isoformat(), playbook_id),
                )
            else:
                conn.execute(
                    "UPDATE research_playbooks SET "
                    "times_applied = times_applied + 1, "
                    "success_rate = CAST(times_succeeded AS REAL) / (times_applied + 1), "
                    "updated_at = ? "
                    "WHERE playbook_id = ?",
                    (datetime.now(timezone.utc).isoformat(), playbook_id),
                )

    # ── Auto-discovery from experience ───────────────────────────────

    def discover_from_experience(
        self,
        experience_store: Any,
        min_occurrences: int = 3,
    ) -> list[ResearchPlaybook]:
        """Mine the ExperienceStore for recurring successful patterns.

        Groups successful changes by (problem_category, change_sequence)
        and creates new playbooks when a pattern recurs ≥ min_occurrences.
        """
        from alphaevo.reflection.experience import ExperienceQuery

        successes = experience_store.query(
            ExperienceQuery(
                only_worked=True,
                limit=200,
                exclude_test_sources=True,
            )
        )
        if len(successes) < min_occurrences:
            return []

        # Group by failure_pattern → change_type sequence
        pattern_actions: dict[str, list[tuple[str, str, str]]] = {}
        for rec in successes:
            for fp in rec.failure_patterns:
                category = _classify_failure_pattern(fp)
                if category:
                    key = category
                    pattern_actions.setdefault(key, []).append(
                        (rec.change_type.value, rec.target, rec.reason)
                    )

        discovered: list[ResearchPlaybook] = []
        for category, actions in pattern_actions.items():
            if len(actions) < min_occurrences:
                continue

            # Count most common action sequences
            from collections import Counter

            action_counts = Counter((a[0], _categorize_target(a[1])) for a in actions)
            top_actions = action_counts.most_common(3)

            steps = []
            for (action, target_cat), _count in top_actions:
                # Find the best reason from matching records
                matching_reasons = [
                    a[2] for a in actions
                    if a[0] == action and _categorize_target(a[1]) == target_cat
                ]
                best_reason = max(matching_reasons, key=len) if matching_reasons else ""
                steps.append(
                    PlaybookStep(
                        action=action,
                        target=target_cat,
                        detail=best_reason or f"Apply {action} to {target_cat}",
                    )
                )

            if not steps:
                continue

            pb_id = f"auto_{category}_{_content_hash(steps)}"
            playbook = ResearchPlaybook(
                playbook_id=pb_id,
                name=f"Auto-discovered: {category} fix",
                trigger=f"Automatically discovered from {len(actions)} successful changes",
                problem_category=category,
                steps=steps,
                success_rate=len(actions) / max(1, len(actions) + 1),  # conservative
                times_applied=len(actions),
                times_succeeded=len(actions),
                source="auto",
            )
            # Don't overwrite builtins
            existing = self.get(pb_id)
            if existing is None:
                self.save(playbook)
                discovered.append(playbook)

        return discovered

    # ── Prompt injection ─────────────────────────────────────────────

    def format_for_prompt(
        self,
        problem_categories: list[str],
        *,
        limit: int = 3,
    ) -> str:
        """Format relevant playbooks as structured guidance for LLM.

        Unlike raw experience dumps, this gives the LLM _procedure_ not just
        _examples_. The LLM can follow or deviate from the playbook.
        """
        playbooks: list[ResearchPlaybook] = []
        for cat in problem_categories:
            playbooks.extend(self.match(cat, limit=limit))

        if not playbooks:
            return ""

        # Deduplicate
        seen: set[str] = set()
        unique: list[ResearchPlaybook] = []
        for pb in playbooks:
            if pb.playbook_id not in seen:
                seen.add(pb.playbook_id)
                unique.append(pb)

        lines = ["### Research Playbooks (proven procedures)"]
        for pb in unique[:limit]:
            success_info = ""
            if pb.times_applied > 0:
                success_info = f" [{pb.times_applied}x applied, {pb.success_rate:.0%} success]"
            lines.append(f"\n**{pb.name}**{success_info}")
            lines.append(f"  Trigger: {pb.trigger}")
            for i, step in enumerate(pb.steps, 1):
                lines.append(f"  Step {i}: {step.detail}")

        return "\n".join(lines)

    # ── internals ────────────────────────────────────────────────────

    def _row_to_playbook(self, row: tuple) -> ResearchPlaybook:
        steps_raw = json.loads(row[4]) if row[4] else []
        steps = [PlaybookStep(**s) for s in steps_raw]
        created = datetime.now(timezone.utc)
        updated = datetime.now(timezone.utc)
        with contextlib.suppress(ValueError, TypeError, IndexError):
            created = datetime.fromisoformat(row[9])
        with contextlib.suppress(ValueError, TypeError, IndexError):
            updated = datetime.fromisoformat(row[10])
        return ResearchPlaybook(
            playbook_id=row[0],
            name=row[1],
            trigger=row[2],
            problem_category=row[3],
            steps=steps,
            success_rate=row[5],
            times_applied=row[6],
            times_succeeded=row[7],
            source=row[8] or "auto",
            created_at=created,
            updated_at=updated,
        )

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


# ── Helpers ──────────────────────────────────────────────────────────


def _classify_failure_pattern(pattern: str) -> str | None:
    """Map a failure pattern string to a problem category.

    Order matters: more specific patterns are checked first to avoid
    ambiguity (e.g. 'low win rate with too many false signals' should
    match 'low_win_rate', not 'low_signals').
    """
    low = pattern.lower()
    # Check win_rate BEFORE signals — "low win rate" takes priority
    if "win" in low and "rate" in low:
        return "low_win_rate"
    if "drawdown" in low or "max_drawdown" in low:
        return "high_drawdown"
    if "overfit" in low or ("train" in low and ("val" in low or "gap" in low)):
        return "overfit"
    if "profit" in low or "p/l" in low or ("ratio" in low and "loss" in low):
        return "low_pl_ratio"
    # Check signals last — broadest match
    if "signal" in low and ("few" in low or "low" in low or "count" in low):
        return "low_signals"
    return None


def _content_hash(steps: list[PlaybookStep]) -> str:
    """Produce a short deterministic hash from playbook steps."""
    key = "|".join(f"{s.action}:{s.target}" for s in steps)
    return hashlib.sha256(key.encode()).hexdigest()[:8]


def _categorize_target(target: str) -> str:
    """Categorize a change target into a high-level category."""
    if "entry.conditions" in target:
        return "entry_condition"
    if "entry.filters" in target:
        return "entry_filter"
    if "entry.logic" in target:
        return "entry_logic"
    if "stop_loss" in target:
        return "stop_loss"
    if "take_profit" in target:
        return "take_profit"
    if "max_holding" in target:
        return "holding_period"
    if "universe" in target:
        return "universe"
    return "other"
