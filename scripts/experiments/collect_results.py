#!/usr/bin/env python3
"""Collect and summarize experiment results into a unified report.

Usage:
    python scripts/experiments/collect_results.py \\
        --evolution results/evolution/ \\
        --ablation results/ablation/ \\
        --output results/summary.md
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment results")
    parser.add_argument("--evolution", type=str, default="results/evolution/")
    parser.add_argument("--ablation", type=str, default="results/ablation/")
    parser.add_argument("--output", type=str, default="results/summary.md")
    args = parser.parse_args()

    lines = ["# AlphaEvo Experiment Results\n"]

    # Evolution results
    evo_dir = Path(args.evolution)
    if evo_dir.exists():
        csv_file = evo_dir / "evolution_results.csv"
        if csv_file.exists():
            lines.append("## Evolution Experiments\n")
            with open(csv_file, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                lines.append("| Strategy | Seed | Champion Score | Improvement | Rounds |")
                lines.append("|----------|------|---------------|-------------|--------|")
                for r in rows:
                    lines.append(
                        f"| {r.get('strategy_id', '')} "
                        f"| {r.get('seed', '')} "
                        f"| {float(r.get('champion_score', 0)):.4f} "
                        f"| {float(r.get('improvement', 0)):+.4f} "
                        f"| {r.get('rounds_completed', '')} |"
                    )
                lines.append("")

    # Ablation results
    abl_dir = Path(args.ablation)
    if abl_dir.exists():
        csv_file = abl_dir / "ablation_results.csv"
        if csv_file.exists():
            lines.append("## Ablation Experiments\n")
            with open(csv_file, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                lines.append("| Configuration | Score | Improvement | Description |")
                lines.append("|--------------|-------|-------------|-------------|")
                for r in rows:
                    if "error" in r and r["error"]:
                        lines.append(f"| {r.get('ablation', '')} | ERROR | — | {r.get('error', '')} |")
                    else:
                        lines.append(
                            f"| {r.get('ablation', '')} "
                            f"| {float(r.get('champion_score', 0)):.4f} "
                            f"| {float(r.get('improvement', 0)):+.4f} "
                            f"| {r.get('description', '')} |"
                        )
                lines.append("")

    # Write output
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Summary written to {output}")


if __name__ == "__main__":
    main()
