"""Reflection analyzer — uses LLM to identify failure patterns and propose changes.

Given a Strategy and its EvaluationReport, the analyzer:
1. Summarizes key performance issues
2. Identifies failure patterns from worst trades
3. Proposes concrete StrategyChange objects for the mutator
"""

from __future__ import annotations

import logging
from enum import Enum
from time import perf_counter

from alphaevo.backtest.indicators import IndicatorRegistry
from alphaevo.core.llm import LLMClient
from alphaevo.models.enums import ChangeType
from alphaevo.models.execution import (
    CandidateExperiment,
    EvaluationReport,
    LLMCallTelemetry,
    LLMReflectionTelemetry,
    ReflectionResult,
    ResearchHypothesis,
    StrategyChange,
)
from alphaevo.models.strategy import Strategy
from alphaevo.strategy.dsl.serializer import StrategySerializer
from alphaevo.strategy.tunable import (
    resolve_tunable_target,
    tune_period_value,
)

logger = logging.getLogger(__name__)

_DIAGNOSIS_TIMEOUT_SECONDS = 45
_EXPERIMENT_TIMEOUT_SECONDS = 45
_LEGACY_REFLECTION_TIMEOUT_SECONDS = 60
_REFLECTION_MAX_RETRIES = 0


def _warn_missing_keys(data: dict, expected: set[str], step: str) -> None:
    """Log a warning when the LLM response is missing expected top-level keys."""
    missing = expected - set(data.keys())
    if missing:
        logger.warning("LLM %s response missing keys: %s", step, missing)


# ── Step 1: Diagnosis prompt — identify root causes ──────────────────

_DIAGNOSIS_SYSTEM = """You are an expert quantitative strategy analyst.
Your job is to diagnose WHY a stock trading strategy underperforms.
Focus on root-cause analysis, not solutions yet.

Think step by step:
1. What are the most significant performance issues?
2. Which entry conditions or exit rules are most likely causing them?
3. In which market regimes does the strategy fail worst?
4. Is the strategy overfitting, under-filtering, or structurally flawed?
5. Is the underlying market hypothesis failing, or is the implementation/execution failing?
"""

_DIAGNOSIS_USER = """## Strategy
```yaml
{strategy_yaml}
```

## Evaluation Results
- Win Rate: {win_rate:.1%}
- Avg Return: {avg_return:.2%}
- Profit/Loss Ratio: {pl_ratio:.2f}
- Max Drawdown: {max_drawdown:.1%}
- Sharpe Ratio: {sharpe:.2f}
- Signal Count: {signal_count}
- Confidence Score: {confidence:.1%}

## Worst Trades (failures)
{failure_summary}
{overfit_section}
{benchmark_section}
{regime_section}
{event_section}
{hypothesis_section}
{history_section}
{experience_section}
{research_section}
## Task
Diagnose the root causes of underperformance. Respond in JSON:
```json
{{
  "root_causes": [
    {{
      "problem": "concise description of the issue",
      "evidence": "what data supports this diagnosis",
      "severity": "high|medium|low"
    }}
  ],
  "diagnosis_summary": "2-3 sentence overall diagnosis",
  "regime_weakness": "which market regime hurts most and why",
  "structural_issues": ["any fundamental design flaws in the strategy"]
}}
```
"""

# ── Step 2: Multi-candidate experiment design prompt ─────────────────

_EXPERIMENT_SYSTEM = """You are an expert quantitative strategy researcher.
Given a diagnosis of a strategy's problems, design {num_candidates} candidate
experiments to fix or improve it. Each experiment should test a DIFFERENT
hypothesis — do not just vary parameter values of the same idea.

Think like a researcher:
- Each experiment should have a clear hypothesis about what's wrong
- Different experiments should attack the problem from different angles
- Estimate the expected outcome of each experiment
- Consider trade-offs: signal count vs. quality, simplicity vs. precision

Rules:
- Each experiment proposes at most {max_changes} concrete changes
- Only use indicators from: {available_indicators}
- DO NOT repeat changes that failed before (see history below)
- Each change must be concrete: specific indicator, specific value
- Rank experiments by how promising they are (priority_score 0.0-1.0)
"""

_EXPERIMENT_USER = """## Diagnosis
{diagnosis_summary}

## Root Causes
{root_causes_text}

## Strategy YAML
```yaml
{strategy_yaml}
```

## Current Metrics
- Win Rate: {win_rate:.1%} | Avg Return: {avg_return:.2%} | Signals: {signal_count}
- Confidence: {confidence:.1%} | Sharpe: {sharpe:.2f}
{hypothesis_section}
{intensity_hint}
{meta_section}
## Task
Design {num_candidates} candidate experiments. Respond in JSON:
```json
{{
  "candidates": [
    {{
      "hypothesis": {{
        "problem": "what this experiment tries to fix",
        "hypothesis": "why this approach should work",
        "expected_outcome": "e.g. +20% signals, -5% win rate",
        "confidence": 0.7
      }},
      "proposed_changes": [
        {{
          "change_type": "tighten_filter|loosen_filter|add_condition|remove_condition|adjust_exit|change_logic|discover_factor",
          "target": "entry.conditions[indicator=xxx].value",
          "from_value": "<current>",
          "to_value": "<proposed>",
          "reason": "why this specific change"
        }}
      ],
      "priority_score": 0.8,
      "rationale": "why this experiment is worth running"
    }}
  ],
  "lesson_learned": "one-sentence key insight"
}}
```
"""

# ── Legacy single-step prompt (fallback) ─────────────────────────────

_REFLECTION_SYSTEM = """You are an expert quantitative strategy analyst.
Your job is to analyze a stock trading strategy's backtest results,
identify why it underperforms, and propose specific improvements.

Rules:
- Propose at most {max_changes} changes per round
- Each change must be concrete (specific indicator, specific value)
- Prefer small, targeted adjustments over radical rewrites
- Consider: entry conditions too loose/tight, exit too early/late, missing filters
- Always explain the reasoning behind each change
- DO NOT repeat changes that have already been tried and failed (see prior rounds below)
- Build on changes that worked in previous rounds
- Only use indicators from the available registry: {available_indicators}
- If train-val gap is high, the strategy may be overfitting — simplify or remove conditions
- If signal count is low, consider loosening filters rather than adding more conditions
- Prefer changes that improve BOTH win rate and consistency, not just one metric
"""

_REFLECTION_USER = """## Strategy
```yaml
{strategy_yaml}
```

## Evaluation Results
- Win Rate: {win_rate:.1%}
- Avg Return: {avg_return:.2%}
- Profit/Loss Ratio: {pl_ratio:.2f}
- Max Drawdown: {max_drawdown:.1%}
- Sharpe Ratio: {sharpe:.2f}
- Signal Count: {signal_count}
- Confidence Score: {confidence:.1%}

## Worst Trades (failures)
{failure_summary}
{overfit_section}
{benchmark_section}
{regime_section}
{event_section}
{hypothesis_section}
{history_section}
{experience_section}
{research_section}
## Task
Analyze the failures and propose improvements. Respond in JSON:
```json
{{
  "failure_patterns": ["pattern1", "pattern2", ...],
  "reflection_summary": "Brief analysis of what's wrong and why",
  "proposed_changes": [
    {{
      "change_type": "tighten_filter|loosen_filter|add_condition|remove_condition|adjust_exit|change_logic|discover_factor",
      "target": "entry.conditions[indicator=xxx].value",
      "from_value": <current>,
      "to_value": <proposed>,
      "reason": "why this change helps"
    }}
  ],
  "lesson_learned": "One-sentence summary of the key insight from this analysis"
}}
```
"""


class ReflectionAnalyzer:
    """Analyze strategy performance and propose improvements via LLM.

    Uses a two-step reasoning process:
    1. Diagnose: identify root causes of underperformance
    2. Design: generate multiple candidate experiments to test different hypotheses

    Falls back to single-step reflection or heuristics when the two-step
    approach fails.
    """

    def __init__(
        self,
        llm: LLMClient,
        max_changes: int = 3,
        num_candidates: int = 3,
    ) -> None:
        self.llm = llm
        self.max_changes = max_changes
        self.num_candidates = num_candidates
        self._serializer = StrategySerializer()

    @classmethod
    def heuristic_only(cls, *, max_changes: int = 3) -> ReflectionAnalyzer:
        """Create an analyzer that only uses heuristic reflection (no LLM).

        This avoids the fragile ``__new__`` pattern and properly initialises
        all instance attributes so that any method can safely access them.
        """
        instance = cls.__new__(cls)
        instance.llm = None  # type: ignore[assignment]
        instance.max_changes = max_changes
        instance.num_candidates = 0
        instance._serializer = StrategySerializer()
        return instance

    def reflect(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        history_text: str = "",
        experience_text: str = "",
        meta_text: str = "",
        research_summary: str = "",
        intensity: float = 1.0,
    ) -> ReflectionResult:
        """Analyze failures and propose changes.

        Uses two-step LLM reasoning (diagnose → design experiments) when
        available, falls back to single-step reflection, then heuristics.

        Args:
            strategy: The strategy that was backtested.
            evaluation: Backtest evaluation results.
            history_text: Formatted text of previous evolution rounds.
            experience_text: Formatted text of past lessons from experience store.
            meta_text: Meta-learning insights from MetaLearner.
            research_summary: Research log summary for continuity across rounds.
            intensity: Mutation intensity multiplier.

        Returns a ReflectionResult with candidates and proposed_changes.
        """
        started_at = perf_counter()
        telemetry = LLMReflectionTelemetry()

        # Heuristic-only mode (no LLM configured)
        if self.llm is None:
            result = self._reflect_heuristic(strategy, evaluation, intensity=intensity)
            telemetry.path = "heuristic_only"
            telemetry.total_duration_ms = int((perf_counter() - started_at) * 1000)
            result.llm_telemetry = telemetry
            return result

        try:
            result = self._reflect_llm_two_step(
                strategy,
                evaluation,
                history_text=history_text,
                experience_text=experience_text,
                meta_text=meta_text,
                research_summary=research_summary,
                intensity=intensity,
                telemetry=telemetry,
            )
            telemetry.path = "two_step"
            telemetry.total_duration_ms = int((perf_counter() - started_at) * 1000)
            result.llm_telemetry = telemetry
            return result
        except Exception as e:
            logger.warning("Two-step LLM reflection failed: %s — trying single-step", e)
            telemetry.fallback_trigger = str(e)
            try:
                result = self._reflect_llm(
                    strategy,
                    evaluation,
                    history_text=history_text,
                    experience_text=experience_text,
                    meta_text=meta_text,
                    research_summary=research_summary,
                    intensity=intensity,
                    telemetry=telemetry,
                )
                telemetry.path = "single_step_fallback"
                telemetry.total_duration_ms = int((perf_counter() - started_at) * 1000)
                result.llm_telemetry = telemetry
                return result
            except Exception as e2:
                logger.warning("Single-step LLM reflection failed: %s — using heuristics", e2)
                if not telemetry.fallback_trigger:
                    telemetry.fallback_trigger = str(e2)
                result = self._reflect_heuristic(strategy, evaluation, intensity=intensity)
                telemetry.path = "heuristic_fallback"
                telemetry.total_duration_ms = int((perf_counter() - started_at) * 1000)
                result.llm_telemetry = telemetry
                return result

    # ── Two-step LLM reflection (diagnose → design) ─────────────────

    def _reflect_llm_two_step(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        history_text: str = "",
        experience_text: str = "",
        meta_text: str = "",
        research_summary: str = "",
        intensity: float = 1.0,
        telemetry: LLMReflectionTelemetry | None = None,
    ) -> ReflectionResult:
        """Two-step LLM reflection: diagnose root causes, then design experiments."""
        context = self._build_context_sections(
            strategy,
            evaluation,
            history_text=history_text,
            experience_text=experience_text,
            research_summary=research_summary,
        )
        strategy_yaml = self._serializer.to_yaml(strategy)
        failure_lines = self._format_failures(evaluation)

        # ── Step 1: Diagnose ──
        diagnosis_messages = [
            {
                "role": "system",
                "content": _DIAGNOSIS_SYSTEM,
            },
            {
                "role": "user",
                "content": _DIAGNOSIS_USER.format(
                    strategy_yaml=strategy_yaml,
                    win_rate=evaluation.overall.win_rate,
                    avg_return=evaluation.overall.avg_return,
                    pl_ratio=evaluation.overall.profit_loss_ratio,
                    max_drawdown=evaluation.overall.max_drawdown,
                    sharpe=evaluation.overall.sharpe_ratio,
                    signal_count=evaluation.overall.signal_count,
                    confidence=evaluation.confidence_score,
                    failure_summary=failure_lines,
                    overfit_section=context["overfit"],
                    benchmark_section=context["benchmark"],
                    regime_section=context["regime"],
                    event_section=context["event"],
                    hypothesis_section=context["hypothesis"],
                    history_section=context["history"],
                    experience_section=context["experience"],
                    research_section=context["research"],
                ),
            },
        ]

        diagnosis_data = self._timed_reflect_json(
            "diagnosis",
            diagnosis_messages,
            temperature=0.3,
            timeout=self._reflection_timeout(_DIAGNOSIS_TIMEOUT_SECONDS),
            max_retries=_REFLECTION_MAX_RETRIES,
            telemetry=telemetry,
        )
        _warn_missing_keys(
            diagnosis_data,
            {"diagnosis_summary", "root_causes"},
            "diagnosis",
        )
        diagnosis_summary = diagnosis_data.get("diagnosis_summary", "")
        root_causes = diagnosis_data.get("root_causes", [])
        structural_issues = diagnosis_data.get("structural_issues", [])

        # Format root causes for step 2
        root_causes_text = ""
        if root_causes:
            lines = []
            for rc in root_causes:
                sev = rc.get("severity", "medium")
                lines.append(f"- [{sev.upper()}] {rc.get('problem', '')}: {rc.get('evidence', '')}")
            root_causes_text = "\n".join(lines)
        if structural_issues:
            root_causes_text += "\n\nStructural issues:\n" + "\n".join(
                f"- {si}" for si in structural_issues
            )

        # ── Step 2: Design experiments ──
        available = ", ".join(IndicatorRegistry.available())

        intensity_hint = ""
        if intensity < 0.6:
            intensity_hint = (
                "\nIMPORTANT: FINE-TUNING round. Make small, conservative adjustments. "
                "Do NOT add or remove conditions."
            )
        elif intensity > 1.3:
            intensity_hint = (
                "\nIMPORTANT: EXPLORATION round. Consider bolder changes — "
                "larger parameter shifts, adding/removing conditions. "
                "Be creative but grounded."
            )

        meta_section = ""
        if meta_text.strip():
            meta_section = f"\n## Meta-Learning Insights\n{meta_text}\n"

        experiment_messages = [
            {
                "role": "system",
                "content": _EXPERIMENT_SYSTEM.format(
                    num_candidates=self.num_candidates,
                    max_changes=self.max_changes,
                    available_indicators=available,
                ),
            },
            {
                "role": "user",
                "content": _EXPERIMENT_USER.format(
                    diagnosis_summary=diagnosis_summary,
                    root_causes_text=root_causes_text or "No specific root causes identified",
                    strategy_yaml=strategy_yaml,
                    win_rate=evaluation.overall.win_rate,
                    avg_return=evaluation.overall.avg_return,
                    signal_count=evaluation.overall.signal_count,
                    confidence=evaluation.confidence_score,
                    sharpe=evaluation.overall.sharpe_ratio,
                    hypothesis_section=context["hypothesis"],
                    intensity_hint=intensity_hint,
                    meta_section=meta_section,
                    num_candidates=self.num_candidates,
                ),
            },
        ]

        temperature = max(0.2, min(0.9, 0.4 + 0.1 * intensity))
        experiment_data = self._timed_reflect_json(
            "experiment_design",
            experiment_messages,
            temperature=temperature,
            timeout=self._reflection_timeout(_EXPERIMENT_TIMEOUT_SECONDS),
            max_retries=_REFLECTION_MAX_RETRIES,
            telemetry=telemetry,
        )
        _warn_missing_keys(experiment_data, {"candidates"}, "experiment")

        # Parse candidates
        candidates = self._parse_candidates(experiment_data.get("candidates", []))

        # Build failure_patterns from root causes
        failure_patterns = [
            rc.get("problem", "") for rc in root_causes if rc.get("problem")
        ] + structural_issues

        # Primary proposed_changes = top candidate's changes (backward compat)
        proposed_changes: list[StrategyChange] = []
        if candidates:
            # Sort by priority, pick the top one
            candidates.sort(key=lambda c: c.priority_score, reverse=True)
            proposed_changes = candidates[0].proposed_changes[: self.max_changes]

        return ReflectionResult(
            strategy_id=strategy.meta.id,
            evaluation_id=evaluation.evaluation_id,
            failure_patterns=failure_patterns,
            proposed_changes=proposed_changes,
            candidates=candidates,
            diagnosis=diagnosis_summary,
            reflection_summary=diagnosis_summary,
        )

    def _reflect_llm(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        history_text: str = "",
        experience_text: str = "",
        meta_text: str = "",
        research_summary: str = "",
        intensity: float = 1.0,
        telemetry: LLMReflectionTelemetry | None = None,
    ) -> ReflectionResult:
        """Single-step LLM reflection (legacy fallback)."""
        context = self._build_context_sections(
            strategy,
            evaluation,
            history_text=history_text,
            experience_text=experience_text,
            research_summary=research_summary,
        )
        strategy_yaml = self._serializer.to_yaml(strategy)
        failure_lines = self._format_failures(evaluation)
        available = ", ".join(IndicatorRegistry.available())

        intensity_hint = ""
        if intensity < 0.6:
            intensity_hint = (
                "\nIMPORTANT: This is a FINE-TUNING round. Make very small, "
                "conservative adjustments (±5% of current values). "
                "Do NOT add or remove conditions."
            )
        elif intensity > 1.3:
            intensity_hint = (
                "\nIMPORTANT: This is an EXPLORATION round. Consider bolder "
                "changes — larger parameter shifts, adding/removing conditions. "
                "Be creative but still grounded in the data."
            )

        meta_section = ""
        if meta_text.strip():
            meta_section = f"\n## Meta-Learning Insights\n{meta_text}\n"

        messages = [
            {
                "role": "system",
                "content": _REFLECTION_SYSTEM.format(
                    max_changes=self.max_changes,
                    available_indicators=available,
                ),
            },
            {
                "role": "user",
                "content": _REFLECTION_USER.format(
                    strategy_yaml=strategy_yaml,
                    win_rate=evaluation.overall.win_rate,
                    avg_return=evaluation.overall.avg_return,
                    pl_ratio=evaluation.overall.profit_loss_ratio,
                    max_drawdown=evaluation.overall.max_drawdown,
                    sharpe=evaluation.overall.sharpe_ratio,
                    signal_count=evaluation.overall.signal_count,
                    confidence=evaluation.confidence_score,
                    failure_summary=failure_lines,
                    overfit_section=context["overfit"],
                    benchmark_section=context["benchmark"],
                    regime_section=context["regime"],
                    event_section=context["event"],
                    hypothesis_section=context["hypothesis"],
                    history_section=context["history"],
                    experience_section=context["experience"],
                    research_section=context["research"],
                )
                + intensity_hint
                + meta_section,
            },
        ]

        temperature = max(0.1, min(0.8, 0.3 + 0.1 * intensity))
        data = self._timed_reflect_json(
            "single_step",
            messages,
            temperature=temperature,
            timeout=self._reflection_timeout(_LEGACY_REFLECTION_TIMEOUT_SECONDS),
            max_retries=_REFLECTION_MAX_RETRIES,
            telemetry=telemetry,
        )

        if not data.get("proposed_changes"):
            logger.warning("LLM response missing proposed_changes field")
        if not data.get("reflection_summary"):
            logger.warning("LLM response missing reflection_summary field")

        changes = self._parse_changes(data.get("proposed_changes", []))

        return ReflectionResult(
            strategy_id=strategy.meta.id,
            evaluation_id=evaluation.evaluation_id,
            failure_patterns=data.get("failure_patterns", []),
            proposed_changes=changes[: self.max_changes],
            reflection_summary=data.get("reflection_summary", ""),
        )

    # ── Shared helpers ───────────────────────────────────────────────

    def _build_context_sections(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        history_text: str = "",
        experience_text: str = "",
        research_summary: str = "",
    ) -> dict[str, str]:
        """Build formatted context sections for LLM prompts."""
        sections: dict[str, str] = {}

        sections["history"] = (
            f"\n## Evolution History (Previous Rounds)\n{history_text}\n"
            if history_text.strip()
            else ""
        )
        sections["experience"] = (
            f"\n## Past Lessons (Cross-Strategy Experience)\n{experience_text}\n"
            if experience_text.strip()
            else ""
        )
        sections["research"] = (
            f"\n## Research Log (Agent Memory)\n{research_summary}\n"
            if research_summary.strip()
            else ""
        )
        assessment = strategy.assess_market_hypothesis(evaluation)
        summary = assessment.summary
        sections["hypothesis"] = (
            "\n## Strategy Hypothesis Lens\n"
            f"- Thesis: {summary.thesis}\n"
            f"- Expected Regimes: {', '.join(summary.expected_regimes) if summary.expected_regimes else 'unspecified'}\n"
            f"- Key Indicators: {', '.join(summary.key_indicators) if summary.key_indicators else 'none'}\n"
            f"- Signal Style: {summary.signal_style}\n"
            f"- Execution Assumption: {summary.execution_assumption}\n"
            f"- Current Assessment: {assessment.status}\n"
            f"- Assessment Rationale: {assessment.rationale}\n"
            f"- Suggested Next Step: {assessment.next_step}\n"
        )

        sections["benchmark"] = ""
        if evaluation.benchmark is not None:
            sections["benchmark"] = (
                "\n## Benchmark Comparison\n"
                f"- Strategy Return: {evaluation.benchmark.strategy_return:.2%}\n"
                f"- Benchmark Return: {evaluation.benchmark.benchmark_return:.2%}\n"
                f"- Excess Return: {evaluation.benchmark.excess_return:+.2%}\n"
            )
            if evaluation.stress_windows is not None:
                sections["benchmark"] += (
                    f"- Stress-Window Pass Rate: {evaluation.stress_windows.pass_rate:.1%}\n"
                    f"- Worst Stress Alpha: {evaluation.stress_windows.worst_alpha:+.2%}\n"
                )

        sections["regime"] = ""
        if evaluation.by_regime or evaluation.regime_holdout is not None:
            regime_lines = ["\n## Regime Analysis"]
            for metric in evaluation.by_regime[:4]:
                regime_lines.append(
                    f"- {metric.regime.value}: win_rate={metric.win_rate:.1%}, "
                    f"avg_return={metric.avg_return:.2%}, signals={metric.signal_count}"
                )
            if evaluation.regime_holdout is not None:
                worst_regime = (
                    evaluation.regime_holdout.worst_regime.value
                    if evaluation.regime_holdout.worst_regime is not None
                    else "n/a"
                )
                regime_lines.append(
                    f"- Holdout pass rate: {evaluation.regime_holdout.pass_rate:.1%}"
                )
                regime_lines.append(
                    f"- Worst regime gap: {evaluation.regime_holdout.worst_gap:.1%} "
                    f"({worst_regime})"
                )
            sections["regime"] = "\n".join(regime_lines) + "\n"

        sections["event"] = ""
        if evaluation.event_context is not None:
            sections["event"] = (
                "\n## Event / News Context\n"
                f"- Relevant Indicators: "
                f"{', '.join(evaluation.event_context.relevant_indicators) if evaluation.event_context.relevant_indicators else 'none'}\n"
                f"- Provider Coverage: {evaluation.event_context.provider_coverage:.1%}\n"
                f"- Proxy-Only Coverage: {evaluation.event_context.proxy_only_coverage:.1%}\n"
            )

        af = evaluation.anti_overfit
        sections["overfit"] = ""
        if af.train_win_rate > 0 or af.val_win_rate > 0:
            overfit_lines = [
                "\n## Generalization Analysis",
                f"- Train Win Rate: {af.train_win_rate:.1%}",
                f"- Validation Win Rate: {af.val_win_rate:.1%}",
                f"- Test Win Rate: {af.test_win_rate:.1%}",
                f"- Train-Val Gap: {af.train_val_gap:.1%}"
                + (" ⚠️ HIGH" if af.train_val_gap > 0.10 else ""),
                f"- Val-Test Gap: {af.val_test_gap:.1%}"
                + (" ⚠️ HIGH" if af.val_test_gap > 0.08 else ""),
                f"- Yearly Consistency: {af.yearly_consistency:.2f}",
            ]
            if af.is_overfit:
                overfit_lines.append(
                    "⚠️ OVERFITTING DETECTED — prioritize simplification "
                    "and removing conditions over adding new ones"
                )
            sections["overfit"] = "\n".join(overfit_lines) + "\n"

        return sections

    def _timed_reflect_json(
        self,
        stage: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        timeout: int,
        max_retries: int,
        telemetry: LLMReflectionTelemetry | None = None,
    ) -> dict:
        """Measure one reflection-stage LLM call and append structured telemetry."""
        started_at = perf_counter()
        error = ""
        success = False
        try:
            data = self.llm.reflect_json(
                messages,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            success = True
            return data
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            if telemetry is not None:
                model_name = getattr(self.llm, "reflect_model", "")
                if not isinstance(model_name, str):
                    model_name = ""
                telemetry.calls.append(
                    LLMCallTelemetry(
                        stage=stage,  # type: ignore[arg-type]
                        model=model_name,
                        timeout_seconds=timeout,
                        max_retries=max_retries,
                        duration_ms=int((perf_counter() - started_at) * 1000),
                        success=success,
                        error=error,
                    )
                )

    def _reflection_timeout(self, cap_seconds: int) -> int:
        """Cap LLM reflection latency so one bad provider call does not stall a round."""
        base_timeout = getattr(self.llm, "timeout", cap_seconds)
        try:
            timeout = int(base_timeout)
        except (TypeError, ValueError):
            timeout = cap_seconds
        return max(10, min(timeout, cap_seconds))

    @staticmethod
    def _normalize_raw_change(raw: dict) -> list[dict]:
        """Map near-miss change verbs into the supported mutation protocol."""
        change_type = str(raw.get("change_type", "")).strip()
        if change_type != "replace_condition":
            return [raw]

        target = str(raw.get("target", "")).strip()
        from_value = raw.get("from_value")
        to_value = raw.get("to_value")
        reason = str(raw.get("reason", "")).strip()

        container = "entry.conditions"
        if "filters" in target:
            container = "entry.filters"

        indicator = ""
        if isinstance(from_value, dict):
            indicator = str(from_value.get("indicator", "")).strip()
        if not indicator and "indicator=" in target:
            indicator = target.split("indicator=", 1)[1].split("]", 1)[0].strip()

        normalized: list[dict] = []
        remove_target = target
        if indicator:
            remove_target = f"{container}[indicator={indicator}]"
        normalized.append(
            {
                "change_type": "remove_condition",
                "target": remove_target,
                "from_value": from_value if from_value is not None else True,
                "to_value": None,
                "reason": reason or "Normalize replace_condition into remove+add",
            }
        )
        if isinstance(to_value, dict):
            normalized.append(
                {
                    "change_type": "add_condition",
                    "target": container,
                    "from_value": None,
                    "to_value": to_value,
                    "reason": reason or "Normalize replace_condition into remove+add",
                }
            )
        return normalized

    @staticmethod
    def _parse_changes(raw_changes: list[dict]) -> list[StrategyChange]:
        """Parse raw JSON change dicts into StrategyChange objects."""
        changes: list[StrategyChange] = []
        for raw in raw_changes:
            for normalized in ReflectionAnalyzer._normalize_raw_change(raw):
                ct = normalized.get("change_type", "")
                try:
                    change_type = ChangeType(ct)
                except ValueError:
                    logger.warning("Unknown change_type: %s, skipping", ct)
                    continue
                changes.append(
                    StrategyChange(
                        change_type=change_type,
                        target=normalized.get("target", ""),
                        from_value=normalized.get("from_value"),
                        to_value=normalized.get("to_value"),
                        reason=normalized.get("reason", ""),
                    )
                )
        return changes

    def _parse_candidates(self, raw_candidates: list[dict]) -> list[CandidateExperiment]:
        """Parse raw JSON candidate dicts into CandidateExperiment objects."""
        candidates: list[CandidateExperiment] = []
        for raw in raw_candidates:
            raw_hyp = raw.get("hypothesis", {})
            hypothesis = ResearchHypothesis(
                problem=raw_hyp.get("problem", ""),
                hypothesis=raw_hyp.get("hypothesis", ""),
                expected_outcome=raw_hyp.get("expected_outcome", ""),
                confidence=float(raw_hyp.get("confidence", 0.5)),
            )
            changes = self._parse_changes(raw.get("proposed_changes", []))
            candidates.append(
                CandidateExperiment(
                    hypothesis=hypothesis,
                    proposed_changes=changes[: self.max_changes],
                    priority_score=float(raw.get("priority_score", 0.0)),
                    rationale=raw.get("rationale", ""),
                )
            )
        return candidates

    def _reflect_heuristic(
        self,
        strategy: Strategy,
        evaluation: EvaluationReport,
        *,
        intensity: float = 1.0,
    ) -> ReflectionResult:
        """Heuristic-based reflection when LLM is unavailable.

        Uses cross-metric reasoning to avoid contradictory actions.
        The ``intensity`` parameter scales step sizes: higher = explore
        wider, lower = fine-tune.
        """
        patterns: list[str] = []
        changes: list[StrategyChange] = []
        m = evaluation.overall
        tunable = strategy.params.tunable if strategy.params else []
        # Scale step multiplier with intensity (default 2 becomes intensity-scaled)
        step_mult = max(0.5, 2 * intensity)

        # ── Cross-metric decision tree ──
        # Instead of independent if/elif, combine signal_count and win_rate
        # to avoid contradictory loosening + tightening actions.

        if m.signal_count < 30 and m.win_rate > 0.55:
            # Few signals but quality is good → fine-tune precision, don't
            # loosen aggressively (we'd dilute a good filter).
            patterns.append(
                f"Few signals ({m.signal_count}) but high win rate ({m.win_rate:.1%}) "
                f"— fine-tuning entry precision rather than loosening broadly"
            )
            # Gently widen the most restrictive filter (smallest step)
            for param in tunable:
                if len(changes) >= 1:  # Only one change for precision tuning
                    break
                current = self._resolve_param(strategy, param.target)
                if current is None:
                    continue
                step = param.step or (param.range[1] - param.range[0]) / 10
                small_step = step * 0.5  # Half-step for precision
                if ".indicator" not in param.target and (
                    "volume_ratio" in param.target or "relative_strength" in param.target
                ):
                    new_val = round(max(param.range[0], current - small_step), 4)
                    if new_val < current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=param.target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Slight loosening from {current} to {new_val} "
                                    f"to capture more trades while preserving quality"
                                ),
                            )
                        )
                elif ".indicator" in param.target:
                    new_val = tune_period_value(
                        current,
                        small_step,
                        param.target,
                        tighten=False,
                        lo=param.range[0],
                        hi=param.range[1],
                    )
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=param.target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Adjust indicator parameter from {current} to {new_val} "
                                    f"to broaden the filter slightly while preserving quality"
                                ),
                            )
                        )
                elif "close_to_ma" in param.target or "deviation" in param.target:
                    new_val = round(min(param.range[1], current + small_step), 4)
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=param.target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Slight tolerance increase from {current} to {new_val} "
                                    f"to capture more trades while preserving quality"
                                ),
                            )
                        )

        elif m.signal_count < 30 and m.win_rate < 0.40:
            # Few signals AND poor quality → loosen entry first, then tighten
            # stop loss to manage risk on the looser filter.
            patterns.append(
                f"Few signals ({m.signal_count}) AND low win rate ({m.win_rate:.1%}) "
                f"— loosening entry to get more data, tightening stop to manage risk"
            )
            for param in tunable:
                if len(changes) >= self.max_changes - 1:
                    break
                target = param.target
                lo, hi = param.range[0], param.range[1]
                step = param.step or (hi - lo) / 10
                current = self._resolve_param(strategy, target)
                if current is None:
                    continue
                if ".indicator" not in target and (
                    "volume_ratio" in target or "relative_strength" in target
                ):
                    new_val = round(max(lo, current - step * step_mult), 4)
                    if new_val < current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Lower threshold from {current} to {new_val} "
                                    f"to capture more trades"
                                ),
                            )
                        )
                elif ".indicator" in target:
                    new_val = tune_period_value(
                        current,
                        step * step_mult,
                        target,
                        tighten=False,
                        lo=lo,
                        hi=hi,
                    )
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Adjust indicator parameter from {current} to {new_val} "
                                    f"to make the filter less restrictive and gather more trades"
                                ),
                            )
                        )
                elif "close_to_ma" in target or "deviation" in target:
                    new_val = round(min(hi, current + step * step_mult), 4)
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Increase tolerance from {current} to {new_val} "
                                    f"for more signals"
                                ),
                            )
                        )
            # Also tighten stop loss to compensate for looser entry
            if (
                strategy.exit.stop_loss.value
                and strategy.exit.stop_loss.value > 0.03
                and len(changes) < self.max_changes
            ):
                changes.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.stop_loss.value",
                        from_value=strategy.exit.stop_loss.value,
                        to_value=round(strategy.exit.stop_loss.value * 0.8, 4),
                        reason="Tighten stop loss to manage risk with looser entry",
                    )
                )

        elif m.signal_count < 30:
            # Few signals, moderate quality → standard loosening
            patterns.append(
                f"Too few signals ({m.signal_count}) — loosening entry to capture more trades"
            )
            for param in tunable:
                if len(changes) >= self.max_changes:
                    break
                target = param.target
                lo, hi = param.range[0], param.range[1]
                step = param.step or (hi - lo) / 10

                current = self._resolve_param(strategy, target)
                if current is None:
                    continue

                if ".indicator" not in target and (
                    "volume_ratio" in target or "relative_strength" in target
                ):
                    new_val = round(max(lo, current - step * step_mult), 4)
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Lower threshold from {current} to {new_val} "
                                    f"to capture more trades"
                                ),
                            )
                        )
                elif ".indicator" in target:
                    new_val = tune_period_value(
                        current,
                        step * step_mult,
                        target,
                        tighten=False,
                        lo=lo,
                        hi=hi,
                    )
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Adjust indicator parameter from {current} to {new_val} "
                                    f"to relax the trend filter and capture more signals"
                                ),
                            )
                        )
                elif "close_to_ma" in target or "deviation" in target:
                    new_val = round(min(hi, current + step * step_mult), 4)
                    if new_val > current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.LOOSEN_FILTER,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Increase tolerance from {current} to {new_val} "
                                    f"for more signals"
                                ),
                            )
                        )
                elif "stop_loss" in target:
                    new_val = round(min(hi, current + step), 4)
                    if new_val > current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.ADJUST_EXIT,
                                target=target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Widen stop loss from {current} to {new_val} "
                                    f"to avoid premature exits"
                                ),
                            )
                        )

        # --- Low win rate with enough signals: tighten or adjust exit ---
        elif m.win_rate < 0.45 and m.signal_count >= 30:
            if m.max_drawdown > 0.15:
                # Low win rate + high drawdown → tighten stop loss first
                patterns.append("Low win rate AND high drawdown — prioritizing risk management")
                if strategy.exit.stop_loss.value and strategy.exit.stop_loss.value > 0.03:
                    changes.append(
                        StrategyChange(
                            change_type=ChangeType.ADJUST_EXIT,
                            target="exit.stop_loss.value",
                            from_value=strategy.exit.stop_loss.value,
                            to_value=round(strategy.exit.stop_loss.value * 0.7, 4),
                            reason="Tighten stop loss to reduce both drawdown and avg loss",
                        )
                    )
            else:
                patterns.append("Low win rate — entry conditions may be too loose")
                if strategy.exit.stop_loss.value and strategy.exit.stop_loss.value > 0.03:
                    changes.append(
                        StrategyChange(
                            change_type=ChangeType.ADJUST_EXIT,
                            target="exit.stop_loss.value",
                            from_value=strategy.exit.stop_loss.value,
                            to_value=round(strategy.exit.stop_loss.value * 0.75, 4),
                            reason="Tighten stop loss to reduce average loss size",
                        )
                    )
            for param in tunable:
                if len(changes) >= self.max_changes:
                    break
                if ".indicator" not in param.target and "volume_ratio" in param.target:
                    current = self._resolve_param(strategy, param.target)
                    if current and current < param.range[1]:
                        new_val = round(min(param.range[1], current + (param.step or 0.1)), 4)
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.TIGHTEN_FILTER,
                                target=param.target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Raise volume requirement from {current} "
                                    f"to {new_val} for higher quality entries"
                                ),
                            )
                        )
                elif ".indicator" in param.target:
                    current = self._resolve_param(strategy, param.target)
                    if current is None:
                        continue
                    new_val = tune_period_value(
                        current,
                        param.step or 1,
                        param.target,
                        tighten=True,
                        lo=param.range[0],
                        hi=param.range[1],
                    )
                    if new_val != current:
                        changes.append(
                            StrategyChange(
                                change_type=ChangeType.TIGHTEN_FILTER,
                                target=param.target,
                                from_value=current,
                                to_value=new_val,
                                reason=(
                                    f"Adjust indicator parameter from {current} to {new_val} "
                                    f"to make the filter more selective"
                                ),
                            )
                        )

        # --- Poor P/L ratio ---
        elif m.profit_loss_ratio < 1.5 and m.avg_return < 0.02:
            patterns.append("Poor profit/loss ratio — winners not big enough")
            tp = strategy.exit.take_profit
            if tp.type == "rr" and tp.value and tp.value < 3.0:
                changes.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.take_profit.value",
                        from_value=tp.value,
                        to_value=round(tp.value + 0.5, 1),
                        reason="Increase R:R target to let winners run",
                    )
                )

        # --- High drawdown (but win rate is OK) ---
        elif m.max_drawdown > 0.15:
            patterns.append("High drawdown — risk management insufficient")
            if strategy.exit.stop_loss.value and strategy.exit.stop_loss.value > 0.03:
                changes.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.stop_loss.value",
                        from_value=strategy.exit.stop_loss.value,
                        to_value=round(strategy.exit.stop_loss.value * 0.8, 4),
                        reason="Tighten stop loss to limit drawdown",
                    )
                )

        # --- Already good: fine-tune exit for improvement ---
        elif m.signal_count >= 30:
            patterns.append("Solid baseline — fine-tuning exit parameters")
            if strategy.exit.max_holding_days and strategy.exit.max_holding_days > 5:
                changes.append(
                    StrategyChange(
                        change_type=ChangeType.ADJUST_EXIT,
                        target="exit.max_holding_days",
                        from_value=strategy.exit.max_holding_days,
                        to_value=strategy.exit.max_holding_days - 2,
                        reason="Reduce holding period to lock in profits faster",
                    )
                )

        if not patterns:
            patterns.append("Strategy performing well — no changes needed")

        # --- Structural: try AND→OR when conditions suppress signal generation ---
        if (
            m.signal_count < 30
            and strategy.entry.logic == "and"
            and len(strategy.entry.conditions) >= 3
            and len(changes) < self.max_changes
        ):
            changes.append(
                StrategyChange(
                    change_type=ChangeType.CHANGE_LOGIC,
                    target="entry.logic",
                    from_value="and",
                    to_value="or",
                    reason=(
                        "Too few signals with AND logic across "
                        f"{len(strategy.entry.conditions)} conditions — try OR"
                    ),
                )
            )
            patterns.append(
                f"Very few signals ({m.signal_count}) with {len(strategy.entry.conditions)} "
                f"AND conditions — switching to OR for more coverage"
            )

        # --- Consult experience store for additional guidance ---
        # If we have fewer than max_changes, try to fill with historically
        # successful changes from the same strategy family.
        if hasattr(self, "_experience_store") and self._experience_store is not None:
            try:
                from alphaevo.reflection.experience import ExperienceQuery

                successful = self._experience_store.query(
                    ExperienceQuery(
                        only_worked=True,
                        limit=5,
                        exclude_test_sources=True,
                    )
                )
                for rec in successful:
                    if len(changes) >= self.max_changes:
                        break
                    # Don't duplicate what we already proposed
                    existing_targets = {c.target for c in changes}
                    if rec.target in existing_targets:
                        continue
                    # Only replay param adjustment changes (safe to replay)
                    if rec.change_type in (
                        ChangeType.TIGHTEN_FILTER,
                        ChangeType.LOOSEN_FILTER,
                        ChangeType.ADJUST_EXIT,
                    ):
                        current = self._resolve_param(strategy, rec.target)
                        if current is not None and rec.to_value is not None:
                            changes.append(
                                StrategyChange(
                                    change_type=rec.change_type,
                                    target=rec.target,
                                    from_value=current,
                                    to_value=rec.to_value,
                                    reason=(f"Replay historically successful change: {rec.lesson}"),
                                )
                            )
            except Exception as e:
                logger.debug("Experience lookup failed: %s", e)

        summary = "; ".join(patterns)
        return ReflectionResult(
            strategy_id=strategy.meta.id,
            evaluation_id=evaluation.evaluation_id,
            failure_patterns=patterns,
            proposed_changes=changes[: self.max_changes],
            reflection_summary=summary,
        )

    @staticmethod
    def _resolve_param(strategy: Strategy, target: str) -> float | int | None:
        """Resolve a tunable param target path to its current value."""
        try:
            value = resolve_tunable_target(strategy, target)
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return value
        except (AttributeError, ValueError, TypeError):
            pass
        return None

    @staticmethod
    def _format_failures(evaluation: EvaluationReport) -> str:
        """Format failure cases for the LLM prompt."""
        if not evaluation.failure_cases:
            return "No failure cases recorded."
        lines: list[str] = []
        for s in evaluation.failure_cases[:10]:
            if s.exit_reason is None:
                exit_reason = None
            elif isinstance(s.exit_reason, Enum):
                exit_reason = s.exit_reason.value
            else:
                exit_reason = s.exit_reason
            base = (
                f"- {s.symbol} {s.signal_date}: "
                f"entry={s.entry_price:.2f} exit={s.exit_price or 0:.2f} "
                f"return={s.return_pct:.2%} reason={exit_reason}"
            )
            if s.indicator_snapshot:
                snap_parts = [
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in sorted(s.indicator_snapshot.items())
                ]
                base += "  indicators={" + ", ".join(snap_parts) + "}"
            lines.append(base)
        return "\n".join(lines)
