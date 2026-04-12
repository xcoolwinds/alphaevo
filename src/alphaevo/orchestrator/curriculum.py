"""Curriculum learning — progressive difficulty for strategy evolution.

Inspired by:
- Curriculum Learning (Bengio et al.): Train on easy examples first
- Self-Paced Learning: Let the learner control difficulty progression
- OpenAI's domain randomization: Gradually increase environment complexity

Strategies learn faster when trained on progressively harder data:
1. Stage 1 (Easy): Clean uptrend data, low volatility, high signal quality
2. Stage 2 (Medium): Mixed trends, moderate volatility, some noise
3. Stage 3 (Hard): Choppy markets, high volatility, regime changes
4. Stage 4 (Reality): Full market data with all conditions

Each stage's champion graduates to the next stage, building robustness
incrementally rather than throwing the strategy into the deep end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphaevo.core.config import AppConfig

logger = logging.getLogger(__name__)


class DifficultyStage(str, Enum):
    """Progressive difficulty levels for curriculum learning."""

    EASY = "easy"  # Clean trending data
    MEDIUM = "medium"  # Mixed with some noise
    HARD = "hard"  # Choppy, high volatility
    REALITY = "reality"  # Full market conditions


@dataclass
class StageConfig:
    """Configuration for a curriculum stage."""

    stage: DifficultyStage
    description: str
    rounds: int = 3  # Evolution rounds at this stage
    max_symbols: int = 15  # Fewer symbols = faster iteration
    # Date range parameters — later stages use wider date ranges
    lookback_days: int = 180  # How far back to look
    min_score_to_graduate: float = 0.3  # Minimum to move to next stage


@dataclass
class CurriculumResult:
    """Result of curriculum learning across all stages."""

    stages_completed: list[str] = field(default_factory=list)
    stage_scores: dict[str, float] = field(default_factory=dict)
    champion_id: str = ""
    champion_score: float = 0.0
    total_rounds: int = 0
    graduated: bool = False  # True if passed all stages


# Default curriculum
DEFAULT_CURRICULUM: list[StageConfig] = [
    StageConfig(
        stage=DifficultyStage.EASY,
        description="Clean trending conditions — learn basic entry/exit timing",
        rounds=3,
        max_symbols=10,
        lookback_days=120,
        min_score_to_graduate=0.25,
    ),
    StageConfig(
        stage=DifficultyStage.MEDIUM,
        description="Mixed markets — develop robustness to noise",
        rounds=3,
        max_symbols=20,
        lookback_days=250,
        min_score_to_graduate=0.30,
    ),
    StageConfig(
        stage=DifficultyStage.HARD,
        description="Volatile conditions — survive drawdowns and regime shifts",
        rounds=3,
        max_symbols=30,
        lookback_days=365,
        min_score_to_graduate=0.35,
    ),
    StageConfig(
        stage=DifficultyStage.REALITY,
        description="Full market reality — validate generalization",
        rounds=4,
        max_symbols=30,
        lookback_days=730,
        min_score_to_graduate=0.40,
    ),
]


class CurriculumEvolution:
    """Evolves strategies through progressively harder stages.

    Instead of immediately testing on full market data (where signal is noisy),
    first trains on easier conditions. This:
    1. Helps the strategy find good entry/exit timing quickly
    2. Avoids early overfit to specific market regimes
    3. Builds robustness incrementally
    4. Reduces wasted computation on hopeless strategies
    """

    def __init__(
        self,
        config: AppConfig,
        curriculum: list[StageConfig] | None = None,
    ) -> None:
        self.config = config
        self.curriculum = curriculum or DEFAULT_CURRICULUM

    def evolve(
        self,
        strategy_id: str,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> CurriculumResult:
        """Run curriculum learning for a strategy.

        The strategy evolves through each difficulty stage.
        Earlier stages produce the seed for later stages.
        """
        from alphaevo.models.enums import EvolutionMethod
        from alphaevo.orchestrator.evolution import EvolutionPipeline

        def _progress(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        result = CurriculumResult()
        current_id = strategy_id

        for stage_config in self.curriculum:
            stage = stage_config.stage
            _progress(
                f"\n{'=' * 50}\n"
                f"📚 Stage: {stage.value.upper()} — {stage_config.description}\n"
                f"{'=' * 50}"
            )

            # Configure date range for this stage
            end = date.today()
            start = end - timedelta(days=stage_config.lookback_days)
            date_range = (start, end)

            # Run evolution at this difficulty level
            pipeline = EvolutionPipeline(self.config)
            try:
                evo_result = pipeline.evolve(
                    current_id,
                    rounds=stage_config.rounds,
                    method=EvolutionMethod.HYBRID,
                    max_symbols=stage_config.max_symbols,
                    date_range=date_range,
                    on_progress=lambda msg: _progress(f"  {msg}"),
                )
            except Exception as e:
                _progress(f"Stage {stage.value} failed: {e}")
                break

            stage_score = evo_result.champion_score
            result.stages_completed.append(stage.value)
            result.stage_scores[stage.value] = stage_score
            result.total_rounds += len(evo_result.rounds)

            _progress(
                f"\n  Stage {stage.value} complete: "
                f"champion={evo_result.champion_id} "
                f"score={stage_score:.1%}"
            )

            # Check graduation criteria
            if stage_score < stage_config.min_score_to_graduate:
                _progress(
                    f"  ❌ Did not graduate (score {stage_score:.1%} "
                    f"< min {stage_config.min_score_to_graduate:.1%})"
                )
                break

            _progress("  ✅ Graduated to next stage!")
            current_id = evo_result.champion_id or current_id
            result.champion_id = current_id
            result.champion_score = stage_score

        result.graduated = len(result.stages_completed) == len(self.curriculum)

        if result.graduated:
            _progress(f"\n🎓 Full curriculum completed! Champion: {result.champion_id}")
        else:
            _progress(
                f"\n📊 Completed {len(result.stages_completed)}/{len(self.curriculum)} stages"
            )

        return result
