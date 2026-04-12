"""Reflection layer — failure analysis and strategy mutation."""

from alphaevo.reflection.analyzer import ReflectionAnalyzer
from alphaevo.reflection.critic import CritiqueVerdict, SelfCritic
from alphaevo.reflection.meta_learner import EvolutionProfile, MetaLearner
from alphaevo.reflection.mutator import MutationError, StrategyMutator
from alphaevo.reflection.playbook import PlaybookStore, ResearchPlaybook

__all__ = [
    "ReflectionAnalyzer",
    "StrategyMutator",
    "MutationError",
    "SelfCritic",
    "CritiqueVerdict",
    "MetaLearner",
    "EvolutionProfile",
    "PlaybookStore",
    "ResearchPlaybook",
]
