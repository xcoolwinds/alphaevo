#!/usr/bin/env python3
"""Run a fixed, shareable AlphaEvo benchmark suite.

This script is the reproducible path documented in the README:
it runs a small set of benchmark strategies with fixed date ranges,
exports research artifacts, and writes one Markdown summary that can be
shared or compared across runs.

Example:
    python scripts/experiments/run_repro_benchmark.py \
        --adapter yfinance \
        --method llm \
        --rounds 3 \
        --output results/repro-benchmark/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))


DEFAULT_STRATEGIES = [
    "rsi_reversion_v1",
    "ma_crossover_v1",
]


def _export_artifacts(output_dir: Path, strategy_id: str, result: Any) -> dict[str, str]:
    """Write human-readable and training-ready artifacts for one benchmark run."""
    from alphaevo.evaluator.reporter import Reporter
    from alphaevo.research_log.trajectory import (
        export_jsonl,
        export_preference_pairs,
        export_sharegpt,
    )

    strategy_dir = output_dir / strategy_id
    strategy_dir.mkdir(parents=True, exist_ok=True)

    evolution_path = strategy_dir / f"{strategy_id}_evolution.md"
    llm_path = strategy_dir / f"{strategy_id}_llm_evidence.md"
    research_path = strategy_dir / f"{strategy_id}_research_report.md"

    evolution_path.write_text(Reporter.evolution_report(result), encoding="utf-8")
    llm_path.write_text(Reporter.llm_evidence_report(result), encoding="utf-8")
    research_path.write_text(Reporter.research_report(result), encoding="utf-8")

    artifacts = {
        "evolution_report": str(evolution_path),
        "llm_evidence_report": str(llm_path),
        "research_report": str(research_path),
    }

    trajectory = getattr(result, "trajectory", None)
    if trajectory is not None and getattr(trajectory, "steps", None):
        traj_dir = strategy_dir / "trajectory"
        jsonl_path = export_jsonl(trajectory, traj_dir / f"{strategy_id}_trajectory.jsonl")
        sharegpt_path = export_sharegpt(trajectory, traj_dir / f"{strategy_id}_sharegpt.jsonl")
        pref_path = export_preference_pairs(
            trajectory,
            traj_dir / f"{strategy_id}_preference.jsonl",
        )
        artifacts.update(
            {
                "trajectory_jsonl": str(jsonl_path),
                "trajectory_sharegpt": str(sharegpt_path),
            }
        )
        if pref_path.exists():
            artifacts["trajectory_preference"] = str(pref_path)

    return artifacts


def _write_summary(output_dir: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> Path:
    """Render a benchmark summary Markdown file."""
    lines = [
        "# AlphaEvo Reproducible Benchmark",
        "",
        f"- Date range: {manifest['start_date']} -> {manifest['end_date']}",
        f"- Method: `{manifest['method']}`",
        f"- Adapter: `{manifest['adapter']}`",
        f"- Rounds: {manifest['rounds']}",
        f"- Max symbols: {manifest['max_symbols']}",
        f"- Seed: {manifest['seed']}",
        "",
        "## Results",
        "",
        "| Strategy | Start Score | Best Score | Improvement | Start Signals | Best Signals | Status |",
        "|----------|-------------|------------|-------------|---------------|--------------|--------|",
    ]

    for row in rows:
        if row.get("error"):
            lines.append(f"| {row['strategy_id']} | ERROR | ERROR | — | — | — | {row['error']} |")
            continue
        lines.append(
            f"| {row['strategy_id']} | {row['start_score']:.1%} | {row['best_score']:.1%} "
            f"| {row['improvement']:+.1%} | {row['start_signals']} | {row['best_signals']} "
            f"| {row['status']} |"
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "Each strategy directory contains:",
        "- `<strategy_id>_evolution.md`",
        "- `<strategy_id>_llm_evidence.md`",
        "- `<strategy_id>_research_report.md`",
        "- `trajectory/*.jsonl` exports for offline training/evaluation",
        "",
        "## Notes",
        "",
        "- This benchmark fixes inputs and output structure, but live data providers and LLMs can still introduce small run-to-run differences.",
        "- Treat these outputs as research artifacts, not trading advice.",
        "",
    ]

    summary_path = output_dir / "benchmark_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a reproducible AlphaEvo benchmark suite")
    parser.add_argument("--adapter", default="yfinance", choices=["yfinance", "akshare", "dsa"])
    parser.add_argument("--method", default="llm", choices=["llm", "hybrid", "param_search"])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max-symbols", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--output", default="results/repro-benchmark/")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start)
    end_date = date.fromisoformat(args.end)
    if start_date > end_date:
        raise SystemExit("--start must be before --end")

    import numpy as np

    from alphaevo.core.config import ConfigManager
    from alphaevo.models.enums import EvolutionMethod
    from alphaevo.orchestrator.evolution import EvolutionPipeline

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "adapter": args.adapter,
        "method": args.method,
        "rounds": args.rounds,
        "max_symbols": args.max_symbols,
        "seed": args.seed,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "strategies": list(args.strategies),
    }
    (output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    config = ConfigManager().load(cli_overrides={"data": {"adapter": args.adapter}})
    method = EvolutionMethod(args.method)
    rows: list[dict[str, Any]] = []

    for strategy_id in args.strategies:
        print(f"\n=== Benchmark: {strategy_id} ({args.method}, {start_date} -> {end_date}) ===")
        pipeline = EvolutionPipeline(config)
        try:
            result = pipeline.evolve(
                strategy_id,
                rounds=args.rounds,
                method=method,
                max_symbols=args.max_symbols,
                date_range=(start_date, end_date),
                on_progress=lambda msg: print(f"  {msg}"),
            )
            best_round = max(
                result.rounds, key=lambda round_result: round_result.evaluation.confidence_score
            )
            first_round = result.rounds[0]
            artifacts = _export_artifacts(output_dir, strategy_id, result)
            row = {
                "strategy_id": strategy_id,
                "start_score": first_round.evaluation.confidence_score,
                "best_score": best_round.evaluation.confidence_score,
                "improvement": result.improvement,
                "start_signals": first_round.evaluation.overall.signal_count,
                "best_signals": best_round.evaluation.overall.signal_count,
                "status": result.stop_reason
                or ("completed" if not result.early_stopped else "early_stopped"),
                "champion_id": result.champion_id,
                "artifacts": artifacts,
            }
            rows.append(row)
            (output_dir / strategy_id / "result.json").write_text(
                json.dumps(row, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            error_row = {"strategy_id": strategy_id, "error": str(exc)}
            rows.append(error_row)
            (output_dir / f"{strategy_id}_error.json").write_text(
                json.dumps(error_row, indent=2),
                encoding="utf-8",
            )
            print(f"  FAILED: {exc}")

    summary_path = _write_summary(output_dir, rows, manifest)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
