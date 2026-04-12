#!/usr/bin/env python3
"""Validate builtin strategies against real market data.

Usage:
    python scripts/validate_real_data.py [--adapter yfinance] [--days 365]

Runs each builtin strategy on real data, generates a standardised report with:
  - Annual return, max drawdown, Sharpe ratio
  - Walk-forward fold metrics
  - CPCV diagnostics
  - Benchmark comparison

Requires a network-accessible data adapter (yfinance or akshare).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ── Setup ──────────────────────────────────────────────────────────────

BUILTIN_DIR = Path(__file__).resolve().parent.parent / "strategies" / "builtin"

# Representative symbols per market — small but diverse
_DEFAULT_SYMBOLS = {
    "yfinance": [
        "AAPL", "MSFT", "GOOG", "AMZN", "NVDA",
        "JPM", "JNJ", "XOM", "PG", "MA",
    ],
    "akshare": [
        "000001", "000002", "600519", "601318", "000858",
        "002415", "600036", "000333", "601012", "000568",
    ],
}


async def _fetch_data(
    adapter_name: str,
    symbols: list[str],
    start: date,
    end: date,
) -> dict[str, pd.DataFrame]:
    """Fetch historical data for the given symbols."""
    from alphaevo.core.config import ConfigManager
    from alphaevo.data.adapter import DataManager
    from alphaevo.data.cache import DataCache

    config = ConfigManager().load()

    if adapter_name == "yfinance":
        from alphaevo.data.adapters.yfinance import YFinanceAdapter
        adapter = YFinanceAdapter()
    elif adapter_name == "akshare":
        from alphaevo.data.adapters.akshare import AkShareAdapter
        adapter = AkShareAdapter()
    else:
        raise ValueError(f"Unknown adapter: {adapter_name}")

    dm = DataManager([adapter], cache=DataCache(config.data.cache_dir))
    data: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = await dm.get_history(sym, start, end)
            if df is not None and not df.empty and len(df) >= 60:
                data[sym] = df
                logger.info("Fetched %s: %d bars", sym, len(df))
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", sym, e)
    return data


def _run_validation(
    adapter_name: str,
    days: int,
    output_dir: Path,
) -> None:
    """Run validation for all builtin strategies."""
    from alphaevo.backtest.engine import BacktestEngine
    from alphaevo.evaluator.metrics import Evaluator
    from alphaevo.models.execution import SampleBatch
    from alphaevo.strategy.dsl.parser import StrategyParser

    end = date.today()
    start = end - timedelta(days=days)
    symbols = _DEFAULT_SYMBOLS.get(adapter_name, _DEFAULT_SYMBOLS["yfinance"])

    print(f"Fetching {len(symbols)} symbols via {adapter_name} ({start} → {end})...")
    data = asyncio.run(_fetch_data(adapter_name, symbols, start, end))
    if not data:
        print("ERROR: No data fetched. Check network and adapter.", file=sys.stderr)
        sys.exit(1)
    print(f"Got data for {len(data)} symbols\n")

    parser = StrategyParser()
    engine = BacktestEngine(slippage=0.001, commission=0.0003, min_data_days=30)
    evaluator = Evaluator()

    yaml_files = sorted(BUILTIN_DIR.glob("*.yaml"))
    if not yaml_files:
        print("ERROR: No strategy files found in", BUILTIN_DIR, file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_lines: list[str] = [
        "# AlphaEvo Real Data Validation Report\n",
        f"**Date**: {date.today()} | **Adapter**: {adapter_name} | "
        f"**Period**: {start} → {end} | **Symbols**: {len(data)}\n",
        "---\n",
    ]

    summary_rows: list[dict] = []

    for yaml_path in yaml_files:
        try:
            strategy = parser.parse_file(yaml_path)
        except Exception as e:
            print(f"  SKIP {yaml_path.name}: parse error ({e})")
            continue

        batch = SampleBatch(
            batch_id=f"validate_{strategy.meta.id}",
            strategy_id=strategy.meta.id,
            symbols=list(data.keys()),
            date_range=(start, end),
        )

        try:
            result = engine.run(strategy, data, batch)
        except Exception as e:
            print(f"  SKIP {strategy.meta.id}: backtest error ({e})")
            continue

        report = evaluator.evaluate(result, strategy, market_data=data)
        m = report.overall

        row = {
            "strategy": strategy.meta.id,
            "signals": result.total_signals,
            "executed": result.executed_signals,
            "win_rate": f"{m.win_rate:.1%}",
            "avg_return": f"{m.avg_return:.2%}",
            "total_return": f"{m.total_return:.2%}",
            "max_dd": f"{m.max_drawdown:.1%}",
            "sharpe": f"{m.sharpe_ratio:.2f}",
            "p_l_ratio": f"{m.profit_loss_ratio:.2f}",
            "confidence": f"{report.confidence_score:.2%}",
        }
        summary_rows.append(row)

        # Per-strategy detail
        report_lines.append(f"## {strategy.meta.name} (`{strategy.meta.id}`)\n")
        report_lines.append(f"- **Signals**: {result.total_signals} ({result.executed_signals} executed)")
        report_lines.append(f"- **Win Rate**: {m.win_rate:.1%}")
        report_lines.append(f"- **Avg Return**: {m.avg_return:.2%}")
        report_lines.append(f"- **Total Return**: {m.total_return:.2%}")
        report_lines.append(f"- **Max Drawdown**: {m.max_drawdown:.1%}")
        report_lines.append(f"- **Sharpe Ratio**: {m.sharpe_ratio:.2f}")
        report_lines.append(f"- **P/L Ratio**: {m.profit_loss_ratio:.2f}")
        report_lines.append(f"- **Confidence Score**: {report.confidence_score:.2%}")

        af = report.anti_overfit
        report_lines.append(f"- **Anti-Overfit**: train_val_gap={af.train_val_gap:.2%}, "
                            f"val_test_gap={af.val_test_gap:.2%}, "
                            f"yearly_consistency={af.yearly_consistency:.2f}")

        if report.walk_forward:
            avg_gap = sum(f.gap for f in report.walk_forward) / len(report.walk_forward)
            report_lines.append(f"- **Walk-Forward**: {len(report.walk_forward)} folds, "
                                f"avg_gap={avg_gap:.2%}")

        if report.cpcv:
            report_lines.append(f"- **CPCV**: {report.cpcv.n_paths} paths, "
                                f"mean_gap={report.cpcv.mean_gap:.2%}, "
                                f"max_gap={report.cpcv.max_gap:.2%}")

        if report.benchmark:
            bm = report.benchmark
            report_lines.append(f"- **Benchmark**: excess={bm.excess_return:.2%}, "
                                f"beats={'YES' if bm.beats_benchmark else 'NO'}")

        report_lines.append("")
        print(f"  {strategy.meta.id}: score={report.confidence_score:.2%}, "
              f"signals={result.total_signals}, win={m.win_rate:.1%}")

    # Summary table
    if summary_rows:
        report_lines.append("---\n## Summary\n")
        header = "| Strategy | Signals | Win Rate | Avg Ret | Total Ret | Max DD | Sharpe | Confidence |"
        sep = "|----------|---------|----------|---------|-----------|--------|--------|------------|"
        report_lines.append(header)
        report_lines.append(sep)
        for r in summary_rows:
            report_lines.append(
                f"| {r['strategy']} | {r['executed']} | {r['win_rate']} | "
                f"{r['avg_return']} | {r['total_return']} | {r['max_dd']} | "
                f"{r['sharpe']} | {r['confidence']} |"
            )
        report_lines.append("")

    report_lines.append("\n---\n*Generated by `scripts/validate_real_data.py`. "
                        "Results are for research purposes only — not financial advice.*\n")

    report_path = output_dir / "validation_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport saved to {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strategies on real data")
    parser.add_argument("--adapter", default="yfinance", choices=["yfinance", "akshare"])
    parser.add_argument("--days", type=int, default=365, help="Lookback period in days")
    parser.add_argument("--output", default="reports", help="Output directory")
    args = parser.parse_args()

    _run_validation(args.adapter, args.days, Path(args.output))


if __name__ == "__main__":
    main()
