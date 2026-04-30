"""Actionable research advice derived from evaluation and optimization results."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from alphaevo.models.execution import EvaluationReport
from alphaevo.models.strategy import Strategy

AdviceStatus = Literal[
    "insufficient_evidence",
    "promote_optimized_exit",
    "optimize_exits",
    "simplify_structure",
    "revise_entry",
    "continue_validation",
]
AdvicePriority = Literal["high", "medium", "low"]


class ResearchRecommendation(BaseModel):
    """One actionable recommendation for the next research step."""

    priority: AdvicePriority
    action: str
    rationale: str
    command: str | None = None


class ResearchAdvice(BaseModel):
    """Structured next-step advice for a strategy research run."""

    strategy_id: str
    status: AdviceStatus
    summary: str
    baseline_score: float
    best_candidate_id: str | None = None
    best_candidate_score: float | None = None
    recommendations: list[ResearchRecommendation] = Field(default_factory=list)


def build_research_advice(
    strategy: Strategy,
    evaluation: EvaluationReport,
    *,
    optimization: Any = None,
    min_signal_count: int = 30,
) -> ResearchAdvice:
    """Build deterministic next-step advice from evaluation evidence."""
    overall = evaluation.overall
    best = getattr(optimization, "best_candidate", None) if optimization is not None else None
    best_score = getattr(getattr(best, "evaluation", None), "confidence_score", None)
    baseline_score = evaluation.confidence_score

    if overall.signal_count < min_signal_count:
        advice = ResearchAdvice(
            strategy_id=strategy.meta.id,
            status="insufficient_evidence",
            summary=(
                f"Only {overall.signal_count} signals fired, below the {min_signal_count}-signal "
                "research threshold. Treat the result as unproven rather than bad."
            ),
            baseline_score=baseline_score,
            best_candidate_id=getattr(best, "candidate_id", None),
            best_candidate_score=best_score,
        )
        advice.recommendations.extend(
            [
                ResearchRecommendation(
                    priority="high",
                    action="Expand the evidence base before judging the thesis.",
                    rationale=(
                        "Small samples can make win rate and average return look better or worse "
                        "than the strategy really is."
                    ),
                    command=f"alphaevo run {strategy.meta.id} --samples 120 --sampling strategy_scoped",
                ),
                ResearchRecommendation(
                    priority="medium",
                    action="Loosen the active buy triggers if a larger sample is still sparse.",
                    rationale=(
                        "Sparse signals usually mean the entry trigger group is too narrow for "
                        "the selected universe or date window."
                    ),
                    command=f'alphaevo strategy revise {strategy.meta.id} "放宽买入触发，保留核心过滤"',
                ),
            ]
        )
        return advice

    if best is not None and isinstance(best_score, float) and best_score >= baseline_score + 0.03:
        has_exit_diagnostics = hasattr(best, "diagnostics")
        optimization_label = "exit/risk" if has_exit_diagnostics else "parameter"
        advice = ResearchAdvice(
            strategy_id=strategy.meta.id,
            status="promote_optimized_exit",
            summary=(
                f"{optimization_label.title()} optimization found a stronger candidate: "
                f"{best.candidate_id} "
                f"({baseline_score:.1%} -> {best_score:.1%})."
            ),
            baseline_score=baseline_score,
            best_candidate_id=best.candidate_id,
            best_candidate_score=best_score,
        )
        advice.recommendations.append(
            ResearchRecommendation(
                priority="high",
                action="Promote the best optimized candidate for another validation run.",
                rationale=(
                    "The improvement came from executable DSL changes on the "
                    "same historical sample, so the next step is validation rather than more search."
                ),
                command=f"alphaevo run {best.candidate_id} --sampling strategy_scoped",
            )
        )
        advice.recommendations.extend(
            _exit_diagnostic_recommendations(getattr(best, "diagnostics", None))
        )
        return advice

    if overall.win_rate >= 0.52 and (overall.avg_return <= 0 or overall.profit_loss_ratio < 1.2):
        advice = ResearchAdvice(
            strategy_id=strategy.meta.id,
            status="optimize_exits",
            summary=(
                "The strategy wins often enough, but payoff quality is weak. "
                "This points to exit timing, stop placement, or take-profit design."
            ),
            baseline_score=baseline_score,
            best_candidate_id=getattr(best, "candidate_id", None),
            best_candidate_score=best_score,
        )
        advice.recommendations.append(
            ResearchRecommendation(
                priority="high",
                action="Focus the next iteration on sell rules and risk/reward, not new entry filters.",
                rationale="Adding more entry conditions may reduce sample size without fixing payoff asymmetry.",
                command=f"alphaevo optimize {strategy.meta.id} --spaces exit,stoploss,takeprofit,holding",
            )
        )
        if best is not None:
            advice.recommendations.extend(
                _exit_diagnostic_recommendations(getattr(best, "diagnostics", None))
            )
        return advice

    if evaluation.anti_overfit.is_overfit or evaluation.anti_overfit.walk_forward_gap > 0.12:
        return ResearchAdvice(
            strategy_id=strategy.meta.id,
            status="simplify_structure",
            summary=(
                "Generalization diagnostics are weak. Prefer simplification before adding more "
                "parameters or filters."
            ),
            baseline_score=baseline_score,
            best_candidate_id=getattr(best, "candidate_id", None),
            best_candidate_score=best_score,
            recommendations=[
                ResearchRecommendation(
                    priority="high",
                    action="Simplify brittle entry rules and rerun walk-forward validation.",
                    rationale=(
                        "Large validation gaps or parameter sensitivity usually mean the strategy "
                        "is fitting the sampled window too tightly."
                    ),
                    command=f'alphaevo strategy revise {strategy.meta.id} "简化过滤条件，降低参数敏感度"',
                )
            ],
        )

    if overall.win_rate < 0.45 or overall.avg_return < 0:
        return ResearchAdvice(
            strategy_id=strategy.meta.id,
            status="revise_entry",
            summary=(
                "The baseline edge is weak after enough trades. Revisit entry triggers or market "
                "regime assumptions before more exit tuning."
            ),
            baseline_score=baseline_score,
            best_candidate_id=getattr(best, "candidate_id", None),
            best_candidate_score=best_score,
            recommendations=[
                ResearchRecommendation(
                    priority="high",
                    action="Search entry thresholds or add a higher-quality guard.",
                    rationale=(
                        "Low win rate or negative average return with enough signals usually means "
                        "the entry setup is admitting noisy trades."
                    ),
                    command=f"alphaevo optimize {strategy.meta.id} --spaces entry,params,indicator",
                )
            ],
        )

    advice = ResearchAdvice(
        strategy_id=strategy.meta.id,
        status="continue_validation",
        summary=(
            "The strategy has enough evidence to keep validating the same thesis. "
            "Prioritize robustness checks over adding complexity."
        ),
        baseline_score=baseline_score,
        best_candidate_id=getattr(best, "candidate_id", None),
        best_candidate_score=best_score,
        recommendations=[
            ResearchRecommendation(
                priority="medium",
                action="Run a larger out-of-sample validation before promoting the strategy.",
                rationale="A promising baseline still needs broader universe and date-window checks.",
                command=f"alphaevo run {strategy.meta.id} --samples 120 --sampling strategy_scoped",
            )
        ],
    )
    if best is not None:
        advice.recommendations.extend(
            _exit_diagnostic_recommendations(getattr(best, "diagnostics", None))
        )
    return advice


def render_research_advice(advice: ResearchAdvice) -> str:
    """Render research advice as Markdown."""
    lines = [
        f"# Research Advice: {advice.strategy_id}",
        "",
        f"- Status: `{advice.status}`",
        f"- Baseline score: {advice.baseline_score:.1%}",
    ]
    if advice.best_candidate_id is not None and advice.best_candidate_score is not None:
        lines.append(
            f"- Best optimized candidate: `{advice.best_candidate_id}` "
            f"({advice.best_candidate_score:.1%})"
        )
    lines.extend(["", "## Summary", "", advice.summary, "", "## Recommended Next Steps", ""])
    for item in advice.recommendations:
        lines.append(f"- **{item.priority.upper()}**: {item.action}")
        lines.append(f"  Rationale: {item.rationale}")
        if item.command:
            lines.append(f"  Command: `{item.command}`")
    lines.extend(["", "*Research output only. Not investment advice.*"])
    return "\n".join(lines)


def _exit_diagnostic_recommendations(diagnostics: Any) -> list[ResearchRecommendation]:
    total = int(getattr(diagnostics, "total_trades", 0) or 0)
    if total <= 0:
        return []

    recs: list[ResearchRecommendation] = []
    sold_early = int(getattr(diagnostics, "sold_early_count", 0) or 0)
    sold_late = int(getattr(diagnostics, "sold_late_count", 0) or 0)
    tp_truncated = int(getattr(diagnostics, "take_profit_truncated_count", 0) or 0)

    if sold_early / total >= 0.25 or tp_truncated / total >= 0.20:
        recs.append(
            ResearchRecommendation(
                priority="medium",
                action="Test a less aggressive take-profit rule or a trailing-profit exit.",
                rationale=(
                    "A meaningful share of exits had further upside shortly after selling, which "
                    "suggests fixed profit-taking may be truncating winners."
                ),
            )
        )
    if sold_late / total >= 0.25:
        recs.append(
            ResearchRecommendation(
                priority="medium",
                action="Tighten the sell trigger or add a trailing stop for winner giveback.",
                rationale=(
                    "The strategy often gave back a meaningful part of intratrade gains before exit."
                ),
            )
        )
    return recs
