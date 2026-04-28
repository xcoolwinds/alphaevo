"""Strategy pattern library — reusable building blocks accumulated from evolution.

Inspired by Voyager (NVIDIA) skill library: as strategies evolve, successful
sub-patterns (entry condition combos, exit configurations, indicator sets)
are extracted and stored for reuse across strategy families.

This enables cross-pollination: a successful entry pattern from a trend
strategy can be suggested when building a reversal strategy.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from alphaevo.models.enums import StrategyCategory
from alphaevo.models.execution import EvaluationReport
from alphaevo.models.strategy import Strategy

# ── Models ──────────────────────────────────────────────────────────


class StrategyPattern(BaseModel):
    """A reusable strategy building block extracted from successful evolution."""

    pattern_id: str
    name: str
    category: StrategyCategory | None = None
    pattern_type: str  # "entry_combo", "exit_config", "indicator_set", "filter_chain"
    description: str = ""
    conditions: list[dict[str, Any]] = Field(default_factory=list)  # Serialized conditions
    exit_config: dict[str, Any] | None = None  # Serialized exit if applicable
    source_strategy: str = ""  # Strategy that originated this pattern
    confidence_score: float = 0.0  # Score of the strategy when pattern was extracted
    win_rate: float = 0.0
    signal_count: int = 0
    times_used: int = 0
    times_succeeded: int = 0

    @property
    def success_rate(self) -> float:
        return self.times_succeeded / self.times_used if self.times_used > 0 else 0.0

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── SQLite Schema ───────────────────────────────────────────────────

_LIBRARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_patterns (
    pattern_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    pattern_type TEXT NOT NULL,
    description TEXT DEFAULT '',
    conditions TEXT DEFAULT '[]',
    exit_config TEXT,
    source_strategy TEXT DEFAULT '',
    confidence_score REAL DEFAULT 0.0,
    win_rate REAL DEFAULT 0.0,
    signal_count INTEGER DEFAULT 0,
    times_used INTEGER DEFAULT 0,
    times_succeeded INTEGER DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_pattern_type ON strategy_patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_pattern_score ON strategy_patterns(confidence_score DESC);
"""


class PatternLibrary:
    """Persistent library of reusable strategy patterns.

    Patterns are extracted from champion strategies and made available
    for future strategy generation and evolution. Tracks usage and
    success rates for each pattern.
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
        self.init_db()

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_LIBRARY_SCHEMA)

    def save(self, pattern: StrategyPattern) -> None:
        """Save or update a pattern."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO strategy_patterns
                    (pattern_id, name, category, pattern_type, description,
                     conditions, exit_config, source_strategy,
                     confidence_score, win_rate, signal_count,
                     times_used, times_succeeded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pattern.pattern_id,
                    pattern.name,
                    pattern.category.value if pattern.category else None,
                    pattern.pattern_type,
                    pattern.description,
                    json.dumps(pattern.conditions),
                    json.dumps(pattern.exit_config) if pattern.exit_config else None,
                    pattern.source_strategy,
                    pattern.confidence_score,
                    pattern.win_rate,
                    pattern.signal_count,
                    pattern.times_used,
                    pattern.times_succeeded,
                    pattern.created_at.isoformat(),
                ),
            )

    def get(self, pattern_id: str) -> StrategyPattern | None:
        """Retrieve a pattern by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM strategy_patterns WHERE pattern_id = ?",
                (pattern_id,),
            ).fetchone()
        return self._row_to_pattern(row) if row else None

    def get_best_patterns(
        self,
        pattern_type: str | None = None,
        category: StrategyCategory | None = None,
        source_family: str | None = None,
        exclude_test_sources: bool = False,
        limit: int = 10,
        min_score: float = 0.3,
    ) -> list[StrategyPattern]:
        """Retrieve top patterns by confidence score."""
        clauses: list[str] = ["confidence_score >= ?"]
        params: list[Any] = [min_score]

        if pattern_type:
            clauses.append("pattern_type = ?")
            params.append(pattern_type)
        if category:
            clauses.append("category = ?")
            params.append(category.value)
        if source_family:
            clauses.append("(source_strategy = ? OR source_strategy LIKE ?)")
            params.extend([source_family, f"{source_family}_v%"])
        if exclude_test_sources:
            clauses.append("source_strategy NOT LIKE 'test_%'")

        where = " AND ".join(clauses)
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM strategy_patterns WHERE {where} "
                f"ORDER BY confidence_score DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_pattern(row) for row in rows]

    def record_usage(self, pattern_id: str, succeeded: bool) -> None:
        """Track pattern usage and whether it led to improvement."""
        with self._connect() as conn:
            if succeeded:
                conn.execute(
                    "UPDATE strategy_patterns SET times_used = times_used + 1, "
                    "times_succeeded = times_succeeded + 1 WHERE pattern_id = ?",
                    (pattern_id,),
                )
            else:
                conn.execute(
                    "UPDATE strategy_patterns SET times_used = times_used + 1 WHERE pattern_id = ?",
                    (pattern_id,),
                )

    def extract_patterns_from_strategy(
        self,
        strategy: Strategy | object,
        evaluation: EvaluationReport | object,
    ) -> list[StrategyPattern]:
        """Extract reusable patterns from a successful strategy.

        Only extracts if the strategy has meaningful signal count and decent score.
        """
        if not isinstance(strategy, Strategy) or not isinstance(evaluation, EvaluationReport):
            return []

        # Only extract patterns from reasonably successful strategies
        if evaluation.overall.signal_count < 20 or evaluation.confidence_score < 0.3:
            return []

        patterns: list[StrategyPattern] = []
        sid = strategy.meta.id
        cat = strategy.meta.category

        entry_rules = [
            *strategy.entry.triggers,
            *strategy.entry.conditions,
            *strategy.entry.guards,
            *strategy.entry.filters,
        ]

        # 1. Extract entry condition combos
        if len(entry_rules) >= 2:
            entry_conds = [
                {"indicator": c.indicator, "op": c.op, "value": c.value}
                for c in entry_rules
            ]
            pattern = StrategyPattern(
                pattern_id=f"entry_{sid}",
                name=f"Entry from {strategy.meta.name}",
                category=cat,
                pattern_type="entry_combo",
                description=f"Entry conditions from {sid} (score={evaluation.confidence_score:.1%})",
                conditions=entry_conds,
                source_strategy=sid,
                confidence_score=evaluation.confidence_score,
                win_rate=evaluation.overall.win_rate,
                signal_count=evaluation.overall.signal_count,
            )
            patterns.append(pattern)

        # 2. Extract exit configuration
        exit_cfg = {
            "stop_loss_type": strategy.exit.stop_loss.type,
            "stop_loss_value": strategy.exit.stop_loss.value,
            "take_profit_type": strategy.exit.take_profit.type,
            "take_profit_value": strategy.exit.take_profit.value,
            "max_holding_days": strategy.exit.max_holding_days,
        }
        exit_pattern = StrategyPattern(
            pattern_id=f"exit_{sid}",
            name=f"Exit from {strategy.meta.name}",
            category=cat,
            pattern_type="exit_config",
            description=f"Exit config from {sid} (P/L={evaluation.overall.profit_loss_ratio:.1f})",
            exit_config=exit_cfg,
            source_strategy=sid,
            confidence_score=evaluation.confidence_score,
            win_rate=evaluation.overall.win_rate,
            signal_count=evaluation.overall.signal_count,
        )
        patterns.append(exit_pattern)

        # 3. Extract indicator combination (which indicators work together)
        indicators = sorted(set(c.indicator for c in entry_rules))
        if len(indicators) >= 2:
            ind_pattern = StrategyPattern(
                pattern_id=f"indicators_{sid}",
                name=f"Indicator set from {strategy.meta.name}",
                category=cat,
                pattern_type="indicator_set",
                description=f"Indicators: {', '.join(indicators)}",
                conditions=[{"indicators": indicators}],
                source_strategy=sid,
                confidence_score=evaluation.confidence_score,
                win_rate=evaluation.overall.win_rate,
                signal_count=evaluation.overall.signal_count,
            )
            patterns.append(ind_pattern)

        return patterns

    def format_for_prompt(
        self,
        category: StrategyCategory | None = None,
        limit: int = 5,
        *,
        exclude_test_sources: bool = False,
    ) -> str:
        """Format top patterns as text for LLM prompt injection."""
        patterns = self.get_best_patterns(
            category=category,
            limit=limit,
            exclude_test_sources=exclude_test_sources,
        )
        if not patterns:
            return ""

        lines: list[str] = ["### Successful Strategy Patterns from Library"]
        for p in patterns:
            usage = (
                f" (used {p.times_used}x, success {p.success_rate:.0%})" if p.times_used > 0 else ""
            )
            lines.append(f"- [{p.pattern_type}] {p.name}: {p.description}{usage}")
            if p.conditions:
                for c in p.conditions[:3]:
                    if "indicator" in c:
                        lines.append(f"    {c['indicator']} {c.get('op', '')} {c.get('value', '')}")
                    elif "indicators" in c:
                        lines.append(f"    Indicators: {', '.join(c['indicators'])}")
        return "\n".join(lines)

    def _row_to_pattern(self, row: tuple) -> StrategyPattern:
        cat = StrategyCategory(row[2]) if row[2] else None
        return StrategyPattern(
            pattern_id=row[0],
            name=row[1],
            category=cat,
            pattern_type=row[3],
            description=row[4] or "",
            conditions=json.loads(row[5]) if row[5] else [],
            exit_config=json.loads(row[6]) if row[6] else None,
            source_strategy=row[7] or "",
            confidence_score=row[8],
            win_rate=row[9],
            signal_count=row[10],
            times_used=row[11],
            times_succeeded=row[12],
        )

    def _connect(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
