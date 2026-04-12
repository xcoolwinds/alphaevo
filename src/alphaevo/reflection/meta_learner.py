"""Meta-learner — learns optimal evolution strategies from accumulated experience.

Inspired by:
- OPRO (DeepMind): Using LLM as optimizer with trajectory history
- Learning to Learn: Meta-RL that adapts the learning process itself
- Bayesian Optimization: Informed exploration of the search space

The meta-learner analyzes the experience store to answer:
- Which change types work best for which problem patterns?
- What's the optimal mutation intensity at each stage of evolution?
- When should we explore vs. exploit?
- Which indicator combinations are most synergistic?
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaevo.models.execution import EvaluationReport
    from alphaevo.reflection.experience import ExperienceStore

logger = logging.getLogger(__name__)


@dataclass
class MetaInsight:
    """A meta-level insight about the evolution process."""

    insight_type: str  # "change_effectiveness", "optimal_intensity", "problem_pattern"
    description: str
    confidence: float = 0.0  # How confident we are in this insight
    data: dict = field(default_factory=dict)


@dataclass
class EvolutionProfile:
    """Recommended evolution parameters based on meta-learning."""

    recommended_method: str = "hybrid"
    recommended_intensity: float = 1.0
    recommended_max_changes: int = 3
    preferred_change_types: list[str] = field(default_factory=list)
    avoid_change_types: list[str] = field(default_factory=list)
    insights: list[MetaInsight] = field(default_factory=list)
    estimated_rounds_to_converge: int = 5


class MetaLearner:
    """Learns from evolution history to optimize future evolution runs.

    Analyzes past evolution experiments to discover:
    1. Which types of changes work in which situations
    2. Optimal intensity schedules
    3. Problem pattern → solution pattern mappings
    4. When to stop (diminishing returns prediction)
    """

    def __init__(self, experience_store: ExperienceStore) -> None:
        self._store = experience_store

    def analyze(
        self,
        family_id: str | None = None,
        evaluation: EvaluationReport | None = None,
    ) -> EvolutionProfile:
        """Generate an evolution profile based on accumulated experience.

        If family_id is provided, makes family-specific recommendations.
        If evaluation is provided, considers current strategy state.
        """
        profile = EvolutionProfile()

        # Gather insights from different analysis dimensions
        change_effectiveness = self._analyze_change_effectiveness(family_id)
        profile.insights.append(change_effectiveness)

        problem_patterns = self._analyze_problem_patterns(family_id)
        profile.insights.append(problem_patterns)

        intensity_insight = self._analyze_optimal_intensity(family_id)
        profile.insights.append(intensity_insight)

        convergence = self._estimate_convergence(family_id)
        profile.insights.append(convergence)

        # Build recommendations from insights
        self._build_recommendations(profile, evaluation)

        return profile

    def _analyze_change_effectiveness(self, family_id: str | None) -> MetaInsight:
        """Determine which change types are most effective."""
        try:
            rates = self._store.get_success_rate_by_change_type(
                family_id=family_id,
                exclude_test_sources=True,
            )
        except Exception as e:
            logger.debug("Change effectiveness analysis failed: %s", e)
            return MetaInsight(
                insight_type="change_effectiveness",
                description="No data available",
            )

        if not rates:
            # Cold start: use builtin playbook priors so the first evolution
            # round gets sensible preferred_change_types instead of empty.
            return self._cold_start_effectiveness(family_id)

        # Rank by success rate (weighted by sample size)
        ranked = sorted(
            rates.items(),
            key=lambda x: x[1]["rate"] * min(1.0, x[1]["total"] / 5),
            reverse=True,
        )

        best = ranked[0] if ranked else None
        worst = ranked[-1] if len(ranked) > 1 else None

        desc_parts = []
        if best:
            desc_parts.append(
                f"Best: {best[0]} ({best[1]['rate']:.0%} success, n={best[1]['total']})"
            )
        if worst and worst[1]["rate"] < 0.3:
            desc_parts.append(
                f"Worst: {worst[0]} ({worst[1]['rate']:.0%} success, n={worst[1]['total']})"
            )

        return MetaInsight(
            insight_type="change_effectiveness",
            description="; ".join(desc_parts) if desc_parts else "Insufficient data",
            confidence=min(1.0, sum(r["total"] for r in rates.values()) / 20),
            data={"rates": rates, "ranked": [r[0] for r in ranked]},
        )

    def _cold_start_effectiveness(self, family_id: str | None) -> MetaInsight:
        """Provide prior change-type effectiveness from builtin playbooks.

        When ExperienceStore has no records yet, we derive a plausible ranking
        from the builtin playbooks' step actions so the first evolution round
        still receives useful preferred/avoid recommendations.
        """
        try:
            from alphaevo.reflection.playbook import _BUILTIN_PLAYBOOKS
        except Exception:
            return MetaInsight(
                insight_type="change_effectiveness",
                description=(
                    "No family-specific evolution history yet"
                    if family_id is not None
                    else "No evolution history yet"
                ),
            )

        # Map playbook step actions → ChangeType names
        action_to_change: dict[str, str] = {
            "tighten_filter": "tighten_filter",
            "loosen_filter": "loosen_filter",
            "add_filter": "add_condition",
            "add_indicator": "add_condition",
            "add_trend_filter": "add_condition",
            "remove_condition": "remove_condition",
            "switch_logic": "change_logic",
            "tighten_stop": "adjust_exit",
            "adjust_exit": "adjust_exit",
            "simplify_exit": "adjust_exit",
            "widen_tp": "adjust_exit",
            "reduce_holding": "adjust_exit",
            "widen_threshold": "loosen_filter",
        }

        counts: dict[str, int] = {}
        for pb in _BUILTIN_PLAYBOOKS:
            for step in pb.steps:
                ct = action_to_change.get(step.action, "")
                if ct:
                    counts[ct] = counts.get(ct, 0) + 1

        if not counts:
            return MetaInsight(
                insight_type="change_effectiveness",
                description="No prior data (cold start)",
            )

        ranked = sorted(counts.items(), key=lambda x: -x[1])
        # Synthesize pseudo-rates: each playbook step is treated as a weak
        # signal (rate=0.5, total=pseudo_count) to bootstrap recommendations.
        rates = {ct: {"rate": 0.50, "total": n} for ct, n in ranked}

        desc = "Cold start prior from builtin playbooks: " + ", ".join(
            f"{ct}(n={n})" for ct, n in ranked[:3]
        )

        return MetaInsight(
            insight_type="change_effectiveness",
            description=desc,
            confidence=0.2,  # Low confidence — it's a prior
            data={"rates": rates, "ranked": [ct for ct, _ in ranked]},
        )

    def _analyze_problem_patterns(self, family_id: str | None) -> MetaInsight:
        """Map problem types to best solutions from history."""
        try:
            from alphaevo.reflection.experience import ExperienceQuery

            records = self._store.query(
                ExperienceQuery(
                    strategy_family=family_id,
                    only_worked=True,
                    limit=50,
                    exclude_test_sources=True,
                )
            )
        except Exception as e:
            logger.debug("Stagnation analysis query failed: %s", e)
            records = []

        if not records:
            return MetaInsight(
                insight_type="problem_pattern",
                description="No successful changes recorded yet",
            )

        # Group successful changes by their lesson content
        pattern_counts: dict[str, int] = defaultdict(int)
        for rec in records:
            # Extract the change pattern (type + target category)
            target_category = _categorize_target(rec.target)
            pattern_key = f"{rec.change_type.value}→{target_category}"
            pattern_counts[pattern_key] += 1

        # Find most common successful patterns
        top_patterns = sorted(pattern_counts.items(), key=lambda x: -x[1])[:5]

        return MetaInsight(
            insight_type="problem_pattern",
            description=f"Top successful patterns: {', '.join(p[0] for p in top_patterns)}",
            confidence=min(1.0, len(records) / 20),
            data={"patterns": dict(top_patterns)},
        )

    def _analyze_optimal_intensity(self, family_id: str | None) -> MetaInsight:
        """Determine optimal mutation intensity from past evolution trajectories."""
        try:
            from alphaevo.reflection.experience import ExperienceQuery

            records = self._store.query(
                ExperienceQuery(
                    strategy_family=family_id,
                    limit=100,
                    exclude_test_sources=True,
                )
            )
        except Exception as e:
            logger.debug("Intensity analysis query failed: %s", e)
            records = []

        if len(records) < 5:
            return MetaInsight(
                insight_type="optimal_intensity",
                description="Insufficient data for intensity analysis",
                data={"recommended": 1.0},
            )

        # Analyze score deltas by round number
        by_round: dict[int, list[float]] = defaultdict(list)
        for rec in records:
            by_round[rec.round_num].append(rec.score_delta)

        # Early rounds (1-2) typically need more intensity, later rounds less
        early_deltas = []
        late_deltas = []
        for rn, deltas in by_round.items():
            if rn <= 2:
                early_deltas.extend(deltas)
            else:
                late_deltas.extend(deltas)

        early_avg = sum(early_deltas) / len(early_deltas) if early_deltas else 0
        late_avg = sum(late_deltas) / len(late_deltas) if late_deltas else 0

        if early_avg > late_avg > 0:
            desc = "Early rounds improve faster — start aggressive, anneal quickly"
            recommended = 1.3
        elif late_avg > early_avg:
            desc = "Late rounds improve more — fine-tuning is effective, start moderate"
            recommended = 0.8
        else:
            desc = "Mixed results — use default intensity schedule"
            recommended = 1.0

        return MetaInsight(
            insight_type="optimal_intensity",
            description=desc,
            confidence=min(1.0, len(records) / 30),
            data={"recommended": recommended, "early_avg": early_avg, "late_avg": late_avg},
        )

    def _estimate_convergence(self, family_id: str | None) -> MetaInsight:
        """Estimate how many rounds until convergence."""
        try:
            from alphaevo.reflection.experience import ExperienceQuery

            records = self._store.query(
                ExperienceQuery(
                    strategy_family=family_id,
                    limit=100,
                    exclude_test_sources=True,
                )
            )
        except Exception as e:
            logger.debug("Max changes analysis query failed: %s", e)
            records = []

        if len(records) < 3:
            return MetaInsight(
                insight_type="convergence",
                description="Insufficient data to estimate convergence",
                data={"estimated_rounds": 5},
            )

        # Find the round where improvements stopped
        last_improvement_round = 0
        for rec in records:
            if rec.worked and rec.score_delta > 0.01:
                last_improvement_round = max(last_improvement_round, rec.round_num)

        estimated = min(10, max(3, last_improvement_round + 2))

        return MetaInsight(
            insight_type="convergence",
            description=f"Improvements typically plateau around round {last_improvement_round}",
            confidence=min(1.0, len(records) / 20),
            data={"estimated_rounds": estimated, "last_improvement": last_improvement_round},
        )

    def _build_recommendations(
        self,
        profile: EvolutionProfile,
        evaluation: EvaluationReport | None,
    ) -> None:
        """Build concrete recommendations from insights."""
        for insight in profile.insights:
            if insight.insight_type == "change_effectiveness":
                ranked = insight.data.get("ranked", [])
                rates = insight.data.get("rates", {})
                if ranked:
                    profile.preferred_change_types = ranked[:3]
                    profile.avoid_change_types = [
                        ct
                        for ct in ranked
                        if ct in rates and rates[ct]["rate"] < 0.2 and rates[ct]["total"] >= 3
                    ]

            elif insight.insight_type == "optimal_intensity":
                profile.recommended_intensity = insight.data.get("recommended", 1.0)

            elif insight.insight_type == "convergence":
                profile.estimated_rounds_to_converge = insight.data.get("estimated_rounds", 5)

        # Current-state-aware adjustments
        if evaluation:
            m = evaluation.overall
            if m.signal_count < 15:
                profile.recommended_intensity = max(1.5, profile.recommended_intensity)
                profile.recommended_method = "hybrid"
                # Few signals → focus on loosening one param at a time
                profile.recommended_max_changes = min(profile.recommended_max_changes, 2)
            elif m.win_rate > 0.6 and m.signal_count > 50:
                profile.recommended_intensity = min(0.7, profile.recommended_intensity)
                profile.recommended_method = "param_search"
                # Good baseline → fine-tune conservatively
                profile.recommended_max_changes = min(profile.recommended_max_changes, 2)
            elif m.signal_count > 100 and m.win_rate < 0.40:
                # Many low-quality signals → tighten aggressively, allow one more
                # but never exceed the default limit (3) to prevent condition stacking
                profile.recommended_max_changes = min(profile.recommended_max_changes + 1, 3)

    def format_meta_insights(self, profile: EvolutionProfile) -> str:
        """Format meta-learning insights as text for LLM prompt."""
        return self.format_meta_insights_static(profile)

    @staticmethod
    def format_meta_insights_static(profile: EvolutionProfile) -> str:
        """Format meta-learning insights as text for LLM prompt (static version)."""
        lines = ["### Meta-Learning Insights"]
        lines.append(f"Recommended intensity: {profile.recommended_intensity:.1f}")
        lines.append(f"Estimated rounds to converge: {profile.estimated_rounds_to_converge}")

        if profile.preferred_change_types:
            lines.append(f"Preferred changes: {', '.join(profile.preferred_change_types)}")
        if profile.avoid_change_types:
            lines.append(f"Avoid: {', '.join(profile.avoid_change_types)}")

        for insight in profile.insights:
            if insight.confidence > 0.2:
                lines.append(f"  [{insight.insight_type}] {insight.description}")

        return "\n".join(lines)


def _categorize_target(target: str) -> str:
    """Categorize a change target into a high-level category."""
    if "entry.conditions" in target:
        return "entry_condition"
    elif "entry.filters" in target:
        return "entry_filter"
    elif "stop_loss" in target:
        return "stop_loss"
    elif "take_profit" in target:
        return "take_profit"
    elif "max_holding" in target:
        return "holding_period"
    elif "universe" in target:
        return "universe"
    return "other"
