"""Island-based evolution — maintain parallel populations for diverse exploration.

Inspired by:
- FunSearch (DeepMind): Island-based evolutionary search with LLM
- OPRO: Optimization by prompting with trajectory history
- MAP-Elites: Quality-diversity optimization

Instead of single-lineage evolution (v1 → v2 → v3), island evolution
maintains multiple parallel lineages ("islands") that independently
evolve and periodically share their best strategies via "migration".

This prevents premature convergence and enables exploration of
fundamentally different strategy approaches simultaneously.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date  # noqa: TC003
from typing import TYPE_CHECKING

from alphaevo.models.enums import EvolutionMethod
from alphaevo.orchestrator.evolution import EvolutionPipeline, EvolutionResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from alphaevo.core.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class IslandState:
    """State of a single evolution island."""

    island_id: int
    strategy_id: str
    result: EvolutionResult | None = None
    champion_score: float = 0.0
    champion_id: str = ""
    generation: int = 0


@dataclass
class IslandEvolutionResult:
    """Result of multi-island evolution."""

    islands: list[IslandState] = field(default_factory=list)
    global_champion_id: str = ""
    global_champion_score: float = 0.0
    total_evaluations: int = 0
    migrations: int = 0

    @property
    def best_island(self) -> IslandState | None:
        if not self.islands:
            return None
        return max(self.islands, key=lambda i: i.champion_score)

    @property
    def diversity_score(self) -> float:
        """How diverse are the island champions (0 = identical, 1 = very different)."""
        if len(self.islands) < 2:
            return 0.0
        scores = [i.champion_score for i in self.islands if i.champion_score > 0]
        if len(scores) < 2:
            return 0.0
        import statistics

        mean = statistics.mean(scores)
        if mean == 0:
            return 0.0
        return min(1.0, statistics.stdev(scores) / mean)


class IslandEvolution:
    """Multi-island evolutionary strategy optimization.

    Architecture:
    - N islands evolve independently for K rounds each
    - After each generation, top strategies "migrate" between islands
    - Each island can use a different evolution method (LLM, param_search, hybrid)
    - Islands with poor performance get "reset" with migrant strategies

    This provides:
    1. Diversity: Multiple approaches explored simultaneously
    2. Robustness: One bad LLM suggestion doesn't derail everything
    3. Cross-pollination: Best ideas spread across islands
    """

    def __init__(
        self,
        config: AppConfig,
        n_islands: int = 3,
        rounds_per_generation: int = 2,
        generations: int = 3,
    ) -> None:
        self.config = config
        self.n_islands = n_islands
        self.rounds_per_gen = rounds_per_generation
        self.generations = generations

    def evolve(
        self,
        strategy_ids: list[str],
        *,
        max_symbols: int = 60,
        date_range: tuple[date, date] | None = None,
        methods: list[EvolutionMethod] | None = None,
        on_progress: Callable[[str], None] | None = None,
    ) -> IslandEvolutionResult:
        """Run island-based evolution across multiple strategy seeds.

        Args:
            strategy_ids: Initial strategies for each island. If fewer than
                n_islands, the first strategy is reused.
            methods: Evolution method per island. Defaults to cycling
                [HYBRID, PARAM_SEARCH, HYBRID].
            max_symbols: Max stocks per backtest.
            date_range: Backtest date range.
            on_progress: Progress callback.
        """

        def _progress(msg: str) -> None:
            logger.info(msg)
            if on_progress:
                on_progress(msg)

        if methods is None:
            methods = [EvolutionMethod.HYBRID, EvolutionMethod.PARAM_SEARCH, EvolutionMethod.HYBRID]

        # Initialize islands
        islands: list[IslandState] = []
        for i in range(self.n_islands):
            sid = strategy_ids[i] if i < len(strategy_ids) else strategy_ids[0]
            islands.append(
                IslandState(
                    island_id=i,
                    strategy_id=sid,
                    champion_id=sid,
                )
            )

        result = IslandEvolutionResult(islands=islands)

        for gen in range(1, self.generations + 1):
            _progress(f"\n{'=' * 50}")
            _progress(f"Generation {gen}/{self.generations}")
            _progress(f"{'=' * 50}")

            # Evolve each island independently
            for island in islands:
                method = methods[island.island_id % len(methods)]
                _progress(
                    f"\n  Island {island.island_id} ({method.value}): "
                    f"evolving {island.strategy_id}..."
                )

                pipeline = EvolutionPipeline(self.config)
                try:
                    evo_result = pipeline.evolve(
                        island.strategy_id,
                        rounds=self.rounds_per_gen,
                        method=method,
                        max_symbols=max_symbols,
                        date_range=date_range,
                        on_progress=lambda msg: _progress(f"    {msg}"),
                    )
                    island.result = evo_result
                    if evo_result.champion_score > island.champion_score:
                        island.champion_score = evo_result.champion_score
                        island.champion_id = evo_result.champion_id or island.strategy_id
                    island.generation = gen
                    result.total_evaluations += len(evo_result.rounds)

                    _progress(
                        f"  Island {island.island_id} champion: "
                        f"{island.champion_id} (score={island.champion_score:.1%})"
                    )
                except Exception as e:
                    _progress(f"  Island {island.island_id} failed: {e}")

            # Migration: share best strategies between islands
            if gen < self.generations:
                self._migrate(islands, _progress)
                result.migrations += 1

        # Find global champion
        best = result.best_island
        if best:
            result.global_champion_id = best.champion_id
            result.global_champion_score = best.champion_score

        _progress(f"\n{'=' * 50}")
        _progress(
            f"Island evolution complete: champion={result.global_champion_id} "
            f"score={result.global_champion_score:.1%} "
            f"evaluations={result.total_evaluations}"
        )
        return result

    def _migrate(
        self,
        islands: list[IslandState],
        _progress: Callable[[str], None],
    ) -> None:
        """Migrate best strategies to weaker islands.

        Top island's champion replaces the worst island's starting point.
        This spreads good genes without losing diversity entirely.
        """
        if len(islands) < 2:
            return

        scored = [(i, i.champion_score) for i in islands if i.champion_score > 0]
        if len(scored) < 2:
            return

        scored.sort(key=lambda x: x[1], reverse=True)
        best_island = scored[0][0]
        worst_island = scored[-1][0]

        # Only migrate if there's a meaningful gap
        gap = best_island.champion_score - worst_island.champion_score
        if gap < 0.05:
            _progress("  Migration skipped: islands are close in performance")
            return

        _progress(
            f"  Migration: Island {best_island.island_id} "
            f"({best_island.champion_id}, {best_island.champion_score:.1%}) "
            f"→ Island {worst_island.island_id} "
            f"(replacing {worst_island.strategy_id})"
        )
        worst_island.strategy_id = best_island.champion_id
