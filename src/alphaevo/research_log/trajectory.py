"""Trajectory export — converts evolution rounds into structured training data.

Inspired by Hermes-Agent's batch_runner + trajectory pipeline: each evolution
round is a clean (hypothesis, diagnosis, changes, outcome) sample that can be
used to:
  1. Fine-tune a reflection/mutation model.
  2. Evaluate prompt effectiveness across runs.
  3. Build offline datasets for RLHF / DPO on strategy research.

Export formats:
  - JSONL: One JSON object per evolution step (hypothesis → outcome).
  - ShareGPT-style: Alternating user/assistant turns for SFT.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Models ──────────────────────────────────────────────────────────


class TrajectoryStep(BaseModel):
    """A single step in an evolution trajectory."""

    round_num: int
    strategy_id: str
    strategy_version: int = 0

    # Input context
    score_before: float
    win_rate_before: float = 0.0
    signal_count_before: int = 0
    failure_patterns: list[str] = Field(default_factory=list)

    # Agent reasoning
    diagnosis: str = ""
    hypothesis: str = ""
    expected_outcome: str = ""

    # Action taken
    changes: list[dict[str, Any]] = Field(default_factory=list)
    method: str = ""  # "llm" | "heuristic" | "param_search"

    # Outcome
    score_after: float = 0.0
    win_rate_after: float = 0.0
    signal_count_after: int = 0
    improved: bool = False
    score_delta: float = 0.0

    # Metadata
    critic_verdict: str = ""
    playbook_used: str = ""
    llm_telemetry: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvolutionTrajectory(BaseModel):
    """Complete trajectory of a strategy evolution session."""

    trajectory_id: str
    strategy_family: str
    initial_score: float = 0.0
    final_score: float = 0.0
    total_rounds: int = 0
    steps: list[TrajectoryStep] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def improvement(self) -> float:
        return self.final_score - self.initial_score

    @property
    def success_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(1 for s in self.steps if s.improved) / len(self.steps)


# ── Collector ───────────────────────────────────────────────────────


class TrajectoryCollector:
    """Collects trajectory data during an evolution run.

    Attach to EvolutionPipeline and call record_step() after each round.
    Call finalize() at the end to produce the complete trajectory.
    """

    def __init__(self, trajectory_id: str, strategy_family: str) -> None:
        self._trajectory_id = trajectory_id
        self._family = strategy_family
        self._steps: list[TrajectoryStep] = []
        self._initial_score: float = 0.0
        self._metadata: dict[str, Any] = {}

    def set_initial_score(self, score: float) -> None:
        self._initial_score = score

    def set_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def record_step(self, step: TrajectoryStep) -> None:
        """Record a single evolution step."""
        self._steps.append(step)

    def finalize(self) -> EvolutionTrajectory:
        """Build the complete trajectory after evolution ends."""
        final_score = self._steps[-1].score_after if self._steps else self._initial_score
        return EvolutionTrajectory(
            trajectory_id=self._trajectory_id,
            strategy_family=self._family,
            initial_score=self._initial_score,
            final_score=final_score,
            total_rounds=len(self._steps),
            steps=self._steps,
            metadata=self._metadata,
        )


# ── Exporters ───────────────────────────────────────────────────────


def export_jsonl(trajectory: EvolutionTrajectory, output_path: Path) -> Path:
    """Export trajectory as JSONL — one line per step.

    Each line is a complete (input, reasoning, action, outcome) record
    suitable for offline analysis or model training.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "a", encoding="utf-8") as f:
        for step in trajectory.steps:
            record = {
                "trajectory_id": trajectory.trajectory_id,
                "strategy_family": trajectory.strategy_family,
                "round": step.round_num,
                "strategy_id": step.strategy_id,
                # Input
                "input": {
                    "score": step.score_before,
                    "win_rate": step.win_rate_before,
                    "signal_count": step.signal_count_before,
                    "failure_patterns": step.failure_patterns,
                },
                # Reasoning
                "reasoning": {
                    "diagnosis": step.diagnosis,
                    "hypothesis": step.hypothesis,
                    "expected_outcome": step.expected_outcome,
                },
                # Action
                "action": {
                    "changes": step.changes,
                    "method": step.method,
                    "critic_verdict": step.critic_verdict,
                    "playbook_used": step.playbook_used,
                    "llm_telemetry": step.llm_telemetry,
                },
                # Outcome
                "outcome": {
                    "score_after": step.score_after,
                    "win_rate_after": step.win_rate_after,
                    "signal_count_after": step.signal_count_after,
                    "improved": step.improved,
                    "score_delta": step.score_delta,
                },
                "timestamp": step.timestamp.isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    logger.info("Exported %d steps to %s", len(trajectory.steps), output_path)
    return output_path


def export_sharegpt(trajectory: EvolutionTrajectory, output_path: Path) -> Path:
    """Export trajectory as ShareGPT-style conversation for SFT.

    Formats each round as a (user_turn, assistant_turn) pair where:
    - user_turn = strategy state + metrics + problem description
    - assistant_turn = diagnosis + hypothesis + proposed changes + outcome
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    conversations: list[dict] = []

    for step in trajectory.steps:
        user_turn = (
            f"Strategy: {step.strategy_id}\n"
            f"Current score: {step.score_before:.1%}\n"
            f"Win rate: {step.win_rate_before:.1%}\n"
            f"Signal count: {step.signal_count_before}\n"
            f"Problems: {'; '.join(step.failure_patterns) if step.failure_patterns else 'none identified'}\n"
            f"\nHow should we improve this strategy?"
        )

        changes_text = "\n".join(
            f"  - {c.get('change_type', '?')}: {c.get('target', '?')} "
            f"({c.get('from_value', '?')} → {c.get('to_value', '?')})"
            for c in step.changes
        ) if step.changes else "  No changes proposed"

        outcome_text = (
            f"Result: {'improved' if step.improved else 'no improvement'} "
            f"(score {step.score_before:.1%} → {step.score_after:.1%}, "
            f"delta={step.score_delta:+.1%})"
        )

        assistant_turn = (
            f"Diagnosis: {step.diagnosis}\n\n"
            f"Hypothesis: {step.hypothesis}\n"
            f"Expected: {step.expected_outcome}\n\n"
            f"Proposed changes:\n{changes_text}\n"
        )
        if step.critic_verdict:
            assistant_turn += f"\nCritic verdict: {step.critic_verdict}\n"
        if step.playbook_used:
            assistant_turn += f"Playbook used: {step.playbook_used}\n"
        if step.llm_telemetry:
            telemetry = step.llm_telemetry
            assistant_turn += (
                "LLM telemetry: "
                f"path={telemetry.get('path', '')}, "
                f"total_duration_ms={telemetry.get('total_duration_ms', 0)}, "
                f"calls={len(telemetry.get('calls', []))}\n"
            )
        assistant_turn += f"\n{outcome_text}"

        conversations.append({
            "conversations": [
                {"from": "human", "value": user_turn},
                {"from": "gpt", "value": assistant_turn},
            ],
            "metadata": {
                "trajectory_id": trajectory.trajectory_id,
                "round": step.round_num,
                "improved": step.improved,
            },
        })

    with open(output_path, "a", encoding="utf-8") as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False, default=str) + "\n")

    logger.info(
        "Exported %d conversation pairs to %s",
        len(conversations),
        output_path,
    )
    return output_path


def export_preference_pairs(
    trajectory: EvolutionTrajectory,
    output_path: Path,
) -> Path:
    """Export (chosen, rejected) preference pairs for DPO/RLHF.

    Generates pairs from rounds where we have both improved and
    non-improved steps, treating improved steps as 'chosen' and
    non-improved as 'rejected'.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    improved_steps = [s for s in trajectory.steps if s.improved]
    failed_steps = [s for s in trajectory.steps if not s.improved]

    if not improved_steps or not failed_steps:
        return output_path

    pairs: list[dict] = []
    for good in improved_steps:
        for bad in failed_steps:
            pair = {
                "trajectory_id": trajectory.trajectory_id,
                "prompt": (
                    f"Strategy: {good.strategy_id}\n"
                    f"Score: {good.score_before:.1%}, "
                    f"Problems: {'; '.join(good.failure_patterns)}"
                ),
                "chosen": (
                    f"Diagnosis: {good.diagnosis}\n"
                    f"Changes: {json.dumps(good.changes, default=str)}\n"
                    f"Result: score {good.score_before:.1%} → {good.score_after:.1%}"
                ),
                "rejected": (
                    f"Diagnosis: {bad.diagnosis}\n"
                    f"Changes: {json.dumps(bad.changes, default=str)}\n"
                    f"Result: score {bad.score_before:.1%} → {bad.score_after:.1%}"
                ),
            }
            pairs.append(pair)

    with open(output_path, "a", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False, default=str) + "\n")

    logger.info("Exported %d preference pairs to %s", len(pairs), output_path)
    return output_path
