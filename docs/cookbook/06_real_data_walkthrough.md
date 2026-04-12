# Real Data Walkthrough

Step-by-step: evolve a strategy with real market data.

## Prerequisites

```bash
pip install alphaevo[data-yfinance]
# Or for A-shares:
pip install alphaevo[data-akshare]
```

## Step 1: Quick Real Data Demo

```bash
# US stocks (AAPL, MSFT, GOOGL, AMZN, META)
alphaevo demo --real

# A-shares (茅台, 五粮液, etc.)
alphaevo demo --real --market cn
```

This runs 3 rounds of heuristic evolution — no LLM needed.

## Step 2: Import a Strategy

```bash
alphaevo strategy import strategies/builtin/ma_crossover.yaml
alphaevo strategy show ma_crossover_v1
```

## Step 3: Run Backtest on Real Data

```bash
alphaevo run ma_crossover_v1 \
    --samples 60 \
    --start 2024-01-01 \
    --end 2025-01-01 \
    --output reports/
```

## Step 4: Evolve with LLM

```bash
export ALPHAEVO_API_KEY=your-key-here

alphaevo evolve rsi_reversion_v1 \
    --rounds 3 \
    --method llm \
    --start 2024-01-01 \
    --end 2025-01-01 \
    --output reports/rsi_evolve/
```

## Step 5: Check Results

```bash
# Leaderboard
alphaevo leaderboard

# Compare original vs evolved
alphaevo compare rsi_reversion_v1 rsi_reversion_v3

# Evolution tree
alphaevo tree rsi_reversion_v1
```

The `--output` directory will now include:

- `<strategy_id>_evolution.md` — round-by-round performance report
- `<strategy_id>_research_report.md` — combined research story with sampling adequacy and LLM appendix
- `<strategy_id>_llm_evidence.md` — focused summary of LLM diagnoses, changes, outcomes, and runtime/fallback telemetry
- `<strategy_id>_research_log.md` — structured research log
- `trajectory/*.jsonl|json` — structured training and evaluation exports

## Step 6: Run Experiments (for paper)

```bash
python scripts/experiments/run_repro_benchmark.py \
    --adapter yfinance \
    --method llm \
    --rounds 3 \
    --output results/repro-benchmark/

python scripts/validate_real_data.py \
    --adapter yfinance \
    --days 365 \
    --output results/real-validation/

python scripts/experiments/run_evolution_experiment.py \
    --strategies ma_crossover_v1 rsi_reversion_v1 \
    --rounds 5 \
    --seeds 42 123 456 \
    --output results/evolution/

python scripts/experiments/run_ablation.py \
    --strategy ma_crossover_v1 \
    --output results/ablation/

python scripts/experiments/collect_results.py \
    --output results/summary.md
```

## Tips

1. **Start with the demo** — verify everything works before using real data
2. **Use `--method llm` when validating model ability** — it isolates the LLM path instead of hiding behind fallback search
3. **Use `hybrid` for practical exploration** — it combines LLM creativity with parameter search precision
4. **Set date ranges** — avoid look-ahead bias, use 1+ year of data
5. **Check for overfitting** — watch the train-val gap in evolution output
6. **min 30 signals** — strategies with too few signals are unreliable

---

*⚠️ This is a research tool. Past backtest performance does not predict future results.*
