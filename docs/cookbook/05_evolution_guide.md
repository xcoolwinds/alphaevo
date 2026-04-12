# Evolution Guide

AlphaEvo's core loop: **backtest → reflect → mutate → retest → repeat**.

## Basic Evolution

```bash
# Evolve for 3 rounds with default hybrid method
alphaevo evolve trend_pullback_rebound_v1 --rounds 3

# Use pure LLM reflection (requires alphaevo[llm])
alphaevo evolve trend_pullback_rebound_v1 --method llm --rounds 5

# Parameter search only (no LLM needed)
alphaevo evolve trend_pullback_rebound_v1 --method param_search
```

## Runtime Controls

AlphaEvo already exposes the main evolution controls at runtime.

```bash
# Control rounds, samples, date range, data source, and models
alphaevo evolve trend_v1 \
    --rounds 5 \
    --method hybrid \
    --samples 50 \
    --start 2025-01-01 \
    --end 2025-12-31 \
    --adapter akshare \
    --model openai/gpt-4o \
    --reflect-model openai/gpt-4o-mini
```

Available controls:

- `--rounds`: evolution rounds
- `--method`: `llm`, `param_search`, or `hybrid`
- `--samples`: max sampled symbols per round
- `--start` / `--end`: backtest window
- `--adapter`: override data adapter
- `--model`: override the main LLM
- `--reflect-model`: override the reflection model

> **Planned**: `--hint` / `--focus` / `--avoid` flags for human-in-the-loop guidance are on the roadmap but not yet implemented in the CLI.

### Island & Curriculum Evolution

These advanced methods are available as both CLI commands and Python API:

```bash
# Multi-island parallel evolution
alphaevo evolve-islands trend_v1 --islands 3 --generations 3 --rounds-per-gen 2

# Curriculum: progressive difficulty (easy → medium → hard → reality)
alphaevo evolve-curriculum trend_v1
```

Or via Python API:

```python
from alphaevo.orchestrator.islands import IslandEvolution
from alphaevo.orchestrator.curriculum import CurriculumEvolution
```

Note: `EvolutionConfig.max_rounds` exists in config, but the current execution path is driven by the explicit CLI/API round arguments above rather than a global hard cap.

## Guided Evolution (Planned)

> **Note**: The `--hint`, `--focus`, and `--avoid` flags are planned but not yet
> implemented in the CLI. Contributions welcome!

Designed usage (once implemented):

```bash
# Suggest a direction
alphaevo evolve trend_v1 --hint "try ATR-based stop loss instead of fixed %"

# Focus on a specific metric
alphaevo evolve trend_v1 --focus "sharpe_ratio"

# Avoid certain directions
alphaevo evolve trend_v1 --avoid "adding more than 5 conditions"

# Combine all three
alphaevo evolve trend_v1 \
    --hint "use volume confirmation" \
    --focus "win_rate" \
    --avoid "increasing complexity"
```

## Evolution Methods

| Method | Requires LLM | Best For |
|--------|-------------|----------|
| `hybrid` | Optional | General use — LLM + parameter search |
| `llm` | Yes | Creative changes — adding/removing conditions |
| `param_search` | No | Fine-tuning — optimizing thresholds |

## Advanced: Multi-Island Evolution (Internal Experimental)

Explore diverse strategies in parallel via Python API:

```python
from alphaevo.orchestrator.islands import IslandEvolution
from alphaevo.core.config import AppConfig

evolution = IslandEvolution(AppConfig())
result = evolution.evolve("trend_v1", islands=3, generations=3, rounds_per_gen=2)
```

Each island evolves independently, sharing best strategies via "migration".

## Advanced: Curriculum Evolution (Internal Experimental)

Progressive difficulty training via Python API:

```python
from alphaevo.orchestrator.curriculum import CurriculumEvolution
from alphaevo.core.config import AppConfig

curriculum = CurriculumEvolution(AppConfig())
result = curriculum.evolve("trend_v1")
```

Stages: Easy → Medium → Hard → Reality.

## Understanding Results

After evolution, you'll see:

The generated evolution artifact now includes the self-improvement context behind each round:

- meta-learning recommendations such as suggested method / intensity / max changes
- family-specific lessons retrieved from the experience store
- reusable strategy patterns injected from the pattern library
- the concrete reflection summary plus approved changes

```
🧬 Evolution Results: trend_pullback_rebound_v1
┌───────┬──────────┬──────────┬────────┬────┬────────┬─────────┬────────┬───────────┐
│ Round │ Strategy │ Win Rate │ Avg Ret│ P/L│ Max DD │ Signals │ Score  │ Status    │
├───────┼──────────┼──────────┼────────┼────┼────────┼─────────┼────────┼───────────┤
│     1 │ trend_v1 │   48.0%  │  1.20% │1.5 │  12.0% │      45 │ 32.5%  │ —         │
│     2 │ trend_v2 │   55.0%  │  2.10% │2.0 │   8.0% │      38 │ 48.2%  │ ✓ improved│
│     3 │ trend_v3 │   61.0%  │  2.80% │2.3 │   7.5% │      35 │ 55.1%  │ ✓ improved│
└───────┴──────────┴──────────┴────────┴────┴────────┴─────────┴────────┴───────────┘
```

## Anti-Overfitting Safeguards

AlphaEvo automatically:
- Splits data into train/val/test
- Detects train-val performance gaps
- Limits strategy complexity (max 8 conditions)
- Checks parameter sensitivity
- Requires ≥30 signals for valid evaluation

## Market / Event Context Caveats

AlphaEvo can already use part of the broader market context:

- market regime metadata and detection
- benchmark-relative indicators
- sector-aware indicators when context is available

But it does not yet have a fully wired news / macro / external-event pipeline.
Many L3 event indicators currently rely on price/volume event proxies when real event data is unavailable.

Practical implication:

- trend / reversal strategies are the strongest current fit
- event / rotation strategies are still more exploratory unless you wire extra data sources

## Capability Priority

For the next stage of the project, the priority is capability depth rather than a Web UI:

- stronger self-evolution memory
- real event/news adapters
- stricter canonical evaluation and walk-forward
- portfolio/risk research on top of single-strategy evolution

See [`../capability_roadmap.md`](../capability_roadmap.md) for the current capability-first roadmap.

See [`../capability_status.md`](../capability_status.md) for the current status summary.

## Evolution Tree

View the full evolution history:

```bash
alphaevo tree trend_pullback_rebound_v1
alphaevo tree --all  # all strategies
```

## Next Steps

- [06 — Real Data Walkthrough](06_real_data_walkthrough.md)
