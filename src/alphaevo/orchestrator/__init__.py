"""Orchestrator — pipeline management for research loops."""

from alphaevo.orchestrator.curriculum import CurriculumEvolution, CurriculumResult
from alphaevo.orchestrator.evolution import EvolutionPipeline, EvolutionResult
from alphaevo.orchestrator.islands import IslandEvolution, IslandEvolutionResult
from alphaevo.orchestrator.pipeline import RunPipeline, RunResult

__all__ = [
    "RunPipeline",
    "RunResult",
    "EvolutionPipeline",
    "EvolutionResult",
    "IslandEvolution",
    "IslandEvolutionResult",
    "CurriculumEvolution",
    "CurriculumResult",
]
