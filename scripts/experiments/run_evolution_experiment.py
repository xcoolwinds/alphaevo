#!/usr/bin/env python3
"""Run multi-strategy evolution experiments.

Usage:
    python scripts/experiments/run_evolution_experiment.py \\
        --strategies trend_pullback_rebound_v1 ma_crossover_v1 \\
        --rounds 5 --seeds 42 123 456 --output results/evolution/

Runs the evolution pipeline on each strategy with each seed,
collecting results for reproducibility analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evolution experiments")
    parser.add_argument(
        "--strategies", nargs="+",
        default=["trend_pullback_rebound_v1", "ma_crossover_v1", "rsi_reversion_v1"],
        help="Strategy IDs to evolve",
    )
    parser.add_argument("--rounds", type=int, default=5, help="Evolution rounds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--output", type=str, default="results/evolution/")
    parser.add_argument("--method", type=str, default="hybrid", choices=["llm", "hybrid", "param_search"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from alphaevo.core.config import ConfigManager
    from alphaevo.models.enums import EvolutionMethod
    from alphaevo.orchestrator.evolution import EvolutionPipeline

    config = ConfigManager().load()
    method = EvolutionMethod(args.method)

    all_results = []

    for strategy_id in args.strategies:
        for seed in args.seeds:
            print(f"\n{'='*60}")
            print(f"Strategy: {strategy_id} | Seed: {seed} | Rounds: {args.rounds}")
            print(f"{'='*60}")

            import random

            import numpy as np
            random.seed(seed)
            np.random.seed(seed)

            pipeline = EvolutionPipeline(config)

            try:
                result = pipeline.evolve(
                    strategy_id,
                    rounds=args.rounds,
                    method=method,
                    on_progress=lambda msg: print(f"  {msg}"),
                )

                row = {
                    "strategy_id": strategy_id,
                    "seed": seed,
                    "rounds_completed": len(result.rounds),
                    "champion_id": result.champion_id or "",
                    "champion_score": result.champion_score,
                    "improvement": result.improvement,
                    "early_stopped": result.early_stopped,
                    "stop_reason": result.stop_reason,
                }

                # Per-round scores
                for r in result.rounds:
                    row[f"round_{r.round_num}_score"] = r.evaluation.confidence_score
                    row[f"round_{r.round_num}_win_rate"] = r.evaluation.overall.win_rate
                    row[f"round_{r.round_num}_signals"] = r.evaluation.overall.signal_count

                all_results.append(row)

                # Save individual result
                result_file = output_dir / f"{strategy_id}_seed{seed}.json"
                result_file.write_text(json.dumps(row, indent=2), encoding="utf-8")

            except Exception as e:
                print(f"  FAILED: {e}")
                all_results.append({
                    "strategy_id": strategy_id,
                    "seed": seed,
                    "error": str(e),
                })

    # Save combined CSV
    if all_results:
        csv_file = output_dir / "evolution_results.csv"
        keys = sorted(set().union(*(r.keys() for r in all_results)))
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nResults saved to {csv_file}")


if __name__ == "__main__":
    main()
