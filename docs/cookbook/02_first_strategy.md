# Creating Your First Strategy

## Option A: From Natural Language (requires LLM)

```bash
pip install alphaevo[llm]
export ALPHAEVO_API_KEY=your-key-here

alphaevo strategy create --market us
# Prompt: "Buy when RSI drops below 30 and price is above 200-day MA"
```

AlphaEvo uses an LLM to convert your idea into a structured YAML DSL.

## Option B: Write YAML Manually

Create a file `my_strategy.yaml`:

```yaml
meta:
  id: my_first_strategy_v1
  name: Simple RSI Strategy
  version: 1
  market: us
  category: reversal
  tags: [rsi, mean-reversion]

description: |
  Buy when RSI-14 drops below 30 (oversold).
  Sell at 4% profit or 3% stop loss.

universe:
  market: [us]

entry:
  logic: and
  conditions:
    - indicator: rsi_14
      op: "<"
      value: 30.0

exit:
  stop_loss:
    type: pct
    value: 0.03
  take_profit:
    type: pct
    value: 0.04
  max_holding_days: 10
```

Import it:

```bash
alphaevo strategy import my_strategy.yaml
alphaevo strategy show my_first_strategy_v1
```

## Option C: Import Built-in Strategies

```bash
# Built-in strategies are loaded automatically on first demo/run
alphaevo demo
alphaevo strategy list
```

## Run Your Strategy

```bash
alphaevo run my_first_strategy_v1 --samples 20
```

## Validate Before Running

```bash
alphaevo strategy validate my_strategy.yaml
```

## Next Steps

- [03 — Understanding the DSL](03_understand_dsl.md)
- [04 — Custom Indicators](04_custom_indicator.md)
