# Trajectory Data Flywheel

AlphaEvo does not only export scores and Markdown reports. Every evolution run
can also emit structured trajectory data that is directly usable for offline
analysis and model training.

## Why it matters

Each round in an evolution session contains a compact research sample:

1. the strategy state before mutation
2. the observed failure patterns
3. the diagnosis and hypothesis
4. the changes that were applied
5. the measured outcome

That makes the trajectory a natural supervision signal for:

- prompt evaluation
- reflection-model fine-tuning
- preference optimization / DPO
- regression testing of agent behavior over time

## Export formats

`alphaevo evolve --output reports/...` and the benchmark scripts can export:

- `trajectory/<strategy_id>_trajectory.jsonl`
  One structured JSON object per evolution step.
- `trajectory/<strategy_id>_sharegpt.jsonl`
  ShareGPT-style conversation turns for SFT workflows.
- `trajectory/<strategy_id>_preference.jsonl`
  Improved vs. non-improved step pairs for preference learning.

## Minimal workflow

```bash
alphaevo evolve rsi_reversion_v1 \
  --method llm \
  --rounds 3 \
  --output reports/rsi_evolve/
```

Then inspect:

- `reports/rsi_evolve/rsi_reversion_v1_research_report.md`
- `reports/rsi_evolve/trajectory/`

## Suggested usage

- Use `trajectory.jsonl` when you want clean per-step analytics.
- Use `sharegpt.jsonl` for instruction-style fine-tuning experiments.
- Use `preference.jsonl` when you want to teach a model which research moves
  led to better outcomes.

## Caveats

- Trajectory data reflects backtest outcomes, not guaranteed live edge.
- Live LLM providers can introduce run-to-run variation.
- Always keep the anti-overfit diagnostics together with the trajectory when
  building downstream datasets.

---

*Research tooling only. Not investment advice.*
