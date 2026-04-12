"""Research Log — structured recording of Agent reasoning during evolution."""

from alphaevo.research_log.context import ContextBuilder, ResearchContext
from alphaevo.research_log.logger import ResearchEvent, ResearchLogger
from alphaevo.research_log.renderer import render_event, render_log_summary, render_round_header
from alphaevo.research_log.trajectory import (
    EvolutionTrajectory,
    TrajectoryCollector,
    TrajectoryStep,
    export_jsonl,
    export_preference_pairs,
    export_sharegpt,
)

__all__ = [
    "ContextBuilder",
    "EvolutionTrajectory",
    "ResearchContext",
    "ResearchEvent",
    "ResearchLogger",
    "TrajectoryCollector",
    "TrajectoryStep",
    "export_jsonl",
    "export_preference_pairs",
    "export_sharegpt",
    "render_event",
    "render_log_summary",
    "render_round_header",
]
