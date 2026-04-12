#!/usr/bin/env python3
"""Run ablation experiments — disable components and measure impact.

Usage:
    python scripts/experiments/run_ablation.py --output results/ablation/

Tests impact of removing: Critic, MetaLearner, Experience Store,
Islands, Curriculum, Alpha Factory.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


ABLATION_CONFIGS = [
    {"name": "full", "description": "All components enabled", "disabled": []},
    {"name": "no_critic", "description": "SelfCritic disabled", "disabled": ["critic"]},
    {"name": "no_meta", "description": "MetaLearner disabled", "disabled": ["meta_learner"]},
    {"name": "no_experience", "description": "ExperienceStore disabled", "disabled": ["experience"]},
    {"name": "no_critic_no_meta", "description": "Critic + Meta disabled", "disabled": ["critic", "meta_learner"]},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation experiments")
    parser.add_argument("--strategy", type=str, default="trend_pullback_rebound_v1")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ablation/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    import random

    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    from alphaevo.core.config import ConfigManager
    from alphaevo.orchestrator.evolution import EvolutionPipeline

    config = ConfigManager().load()
    results = []

    for ablation in ABLATION_CONFIGS:
        print(f"\n{'='*60}")
        print(f"Ablation: {ablation['name']} — {ablation['description']}")
        print(f"{'='*60}")

        random.seed(args.seed)
        np.random.seed(args.seed)

        pipeline = EvolutionPipeline(config)

        # Apply ablation by neutering components
        if "critic" in ablation["disabled"]:
            pipeline._critic = _NullCritic()
        if "meta_learner" in ablation["disabled"]:
            pipeline._meta_learner = _NullMetaLearner()
        if "experience" in ablation["disabled"]:
            pipeline._experience_store = _NullExperienceStore()

        try:
            result = pipeline.evolve(
                args.strategy,
                rounds=args.rounds,
                on_progress=lambda msg: print(f"  {msg}"),
            )

            row = {
                "ablation": ablation["name"],
                "description": ablation["description"],
                "champion_score": result.champion_score,
                "improvement": result.improvement,
                "rounds_completed": len(result.rounds),
                "early_stopped": result.early_stopped,
            }
            for r in result.rounds:
                row[f"round_{r.round_num}_score"] = r.evaluation.confidence_score
            results.append(row)

        except Exception as e:
            print(f"  FAILED: {e}")
            results.append({
                "ablation": ablation["name"],
                "error": str(e),
            })

    # Save CSV
    if results:
        csv_file = output_dir / "ablation_results.csv"
        keys = sorted(set().union(*(r.keys() for r in results)))
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_file}")

    # Print summary table
    print(f"\n{'Ablation':<25} {'Score':>10} {'Improvement':>15}")
    print("-" * 55)
    for r in results:
        if "error" not in r:
            print(f"{r['ablation']:<25} {r['champion_score']:>10.4f} {r['improvement']:>+15.4f}")


class _NullCritic:
    """Critic that approves everything."""
    def critique(self, strategy, evaluation, reflection):
        from alphaevo.reflection.critic import CritiqueVerdict
        v = CritiqueVerdict()
        v.approved = reflection.proposed_changes
        return v


class _NullMetaLearner:
    """MetaLearner that returns default recommendations."""
    def analyze(self, **kwargs):
        from alphaevo.reflection.meta_learner import EvolutionProfile
        return EvolutionProfile()


class _NullExperienceStore:
    """ExperienceStore that stores nothing."""
    def record_batch(self, records): pass
    def query(self, **kwargs): return []
    def get_lessons_for_family(self, family_id, **kwargs): return []


if __name__ == "__main__":
    main()
