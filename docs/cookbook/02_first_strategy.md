# Creating Your First Strategy

## Option A: From Natural Language (no LLM required)

```bash
alphaevo strategy draft "Buy when RSI drops below 30 and price is above 200-day MA; sell when RSI recovers above 65" \
  --market us \
  --save
```

AlphaEvo uses a deterministic strategy drafter to convert concise strategy
ideas into executable YAML. For broader creative generation, use
`alphaevo strategy create` with `alphaevo[llm]`.

To draft, save, backtest, and run entry-parameter plus exit optimization in one command:

```bash
alphaevo strategy research "Buy when RSI drops below 30; sell when price breaks MA10; stop loss 3%; hold 5 days" \
  --market us \
  --adapter yfinance \
  --samples 40
```

After reviewing the result, use `strategy improve` to apply a bounded revision
and immediately validate the revised strategy:

```bash
alphaevo strategy improve my_first_strategy_v1 "reduce drawdown, add right-side confirmation, keep fewer trades" \
  --samples 40 \
  --optimize-params \
  --optimize-exits
```

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
  Buy when RSI-14 drops below 30 (oversold) while price stays above the 200-day MA.
  Sell when RSI recovers above 65, at 4% profit, or at 3% stop loss.

universe:
  market: [us]

entry:
  logic: and
  triggers:
    - indicator: rsi_14
      op: "<"
      value: 30.0
  guards:
    - indicator: close_above_ma200
      op: "=="
      value: true

exit:
  triggers:
    - indicator: rsi_14
      op: ">"
      value: 65
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
