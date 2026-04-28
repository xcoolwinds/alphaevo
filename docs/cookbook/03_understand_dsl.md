# Understanding the Strategy DSL

AlphaEvo strategies are defined in YAML with two layers:
a human-readable description and an executable DSL.

## Structure Overview

```yaml
meta:         # Identity and metadata
description:  # Natural language explanation
universe:     # Which stocks to consider
entry:        # When to buy
exit:         # When to sell
market_rules: # Market-specific rules (T+1, limit up/down)
params:       # Tunable parameter ranges
```

## Meta Section

```yaml
meta:
  id: trend_pullback_v1        # Unique identifier
  name: Trend Pullback Rebound  # Human name
  version: 1                    # Version number
  parent_id: null               # Evolution parent (null = root)
  market: a_share               # a_share / us / hk
  category: trend               # trend / reversal / event / rotation
  tags: [趋势, 回踩]
  preferred_regime:             # Optional: market regimes this works in
    - trending_up
```

## Entry Conditions

```yaml
entry:
  logic: and                    # and | or — how to combine buy triggers
  triggers:                     # Actual buy signals; falls back to conditions if empty
    - indicator: rsi_14         # Must be a registered indicator
      op: "<"                   # ==, !=, <, <=, >, >=
      value: 30.0
  guards:                       # Hard filters, always AND
    - indicator: ma5_above_ma10
      op: "=="
      value: true
  conditions: []                # Backward-compatible legacy trigger field
  filters:                      # Backward-compatible legacy guard field
    - indicator: negative_news_score
      op: "<"
      value: 0.4
  execution:
    timing: next_open           # next_open | close
    slippage: 0.001
```

Prefer `triggers` for the event that should open a position and `guards` for
market-quality or risk filters. Older strategies using `conditions` and
`filters` still run unchanged: if `triggers` is empty, AlphaEvo uses
`conditions` as the buy trigger group; `guards` and `filters` are both applied
as hard AND filters.

## Exit Rules

```yaml
exit:
  triggers:
    - indicator: close_below_ma10
      op: "=="
      value: true               # explicit sell signal, exits at current close
  stop_loss:
    type: pct                   # pct | atr | composite
    value: 0.04                 # 4% stop loss
  take_profit:
    type: rr                    # rr (risk-reward) | pct | trailing
    value: 2.0                  # 2:1 reward/risk
  max_holding_days: 10
```

`exit.triggers` is for explicit sell logic such as "sell when price breaks MA10".
It is evaluated while a position is open. If any trigger passes, the trade exits
with `exit_reason=signal`. Stop loss and take profit are checked before these
close-based sell triggers.

If stop loss and take profit can both be touched in the same candle, AlphaEvo
uses `backtest.fill_policy`: `conservative` assumes the stop fills first,
`optimistic` assumes take profit fills first, and `close_first` resolves by the
bar close relative to the entry price.

### Composite Stop Loss

```yaml
exit:
  stop_loss:
    type: composite
    conditions:
      - indicator: close_below_ma10
        op: "=="
        value: true
      - indicator: rsi_14
        op: ">"
        value: 75
```

## Available Indicators

Run to see all registered indicators:

```python
from alphaevo.backtest.indicators import IndicatorRegistry
print(IndicatorRegistry.available())
```

### L1 (OHLCV only — always available)

`rsi_14`, `ma5_above_ma10`, `close_to_ma10_pct`, `close_above_ma20`,
`volume_ratio_1d_5d`, `atr`, `macd_histogram`,
`macd_histogram_fast12_slow26_signal9`, `macd_cross_bullish`,
`bollinger_band_width`, `bollinger_band_width_20d_std1p5`,
`ma20_slope`, `momentum_10d`, etc.

### L2 (needs benchmark data)

`relative_strength_20d`, `sector_heat_rank` — degrade gracefully when unavailable.

### L3 (needs external APIs)

`negative_news_score`, `news_sentiment_score` — prefer external feeds, otherwise fall back to price/volume event proxies.

## Tunable Parameters

```yaml
params:
  tunable:
    - target: entry.triggers[indicator=rsi_14].value
      range: [20, 40]
      step: 5
    - target: exit.stop_loss.value
      range: [0.02, 0.08]
      step: 0.01
```

These ranges are used by `param_search` evolution and parameter sensitivity analysis.

## Next Steps

- [04 — Custom Indicators](04_custom_indicator.md)
- [05 — Evolution Guide](05_evolution_guide.md)
