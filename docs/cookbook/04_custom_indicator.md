# Custom Indicators

AlphaEvo supports two ways to add indicators.

## Method 1: Static Registration (Python Decorator)

Add to `src/alphaevo/backtest/indicators.py`:

```python
@IndicatorRegistry.register("my_custom_indicator")
def my_custom_indicator(df: pd.DataFrame, idx: int) -> float:
    """My custom indicator description."""
    if idx < 20:
        return 0.0
    window = df["close"].iloc[idx - 19 : idx + 1]
    return float(window.std() / window.mean())  # coefficient of variation
```

### Rules:
- Function signature: `(df: pd.DataFrame, idx: int) -> float | bool`
- `df` has columns: `open`, `high`, `low`, `close`, `volume`
- `idx` is the current bar (0-based) — look back only
- Return `float` or `bool`
- Handle edge cases (idx < lookback → return 0)

## Method 2: Dynamic Registration (Alpha Factory)

LLM-generated factors can be registered at runtime:

```python
from alphaevo.backtest.indicators import IndicatorRegistry

def vol_momentum(df, idx):
    if idx < 10:
        return 0.0
    recent = df["volume"].iloc[idx - 4 : idx + 1].mean()
    older = df["volume"].iloc[idx - 9 : idx - 4].mean()
    return float(recent / older) if older > 0 else 0.0

IndicatorRegistry.register_dynamic("vol_momentum", vol_momentum)

# Verify
assert IndicatorRegistry.is_registered("vol_momentum")
print(IndicatorRegistry.dynamic_names())  # ["vol_momentum"]

# Remove when done
IndicatorRegistry.unregister_dynamic("vol_momentum")
```

## Method 3: Alpha Factory (Automated)

The Alpha Factory module uses LLM to invent and validate new factors:

```python
from alphaevo.alpha_factory import AlphaFactory

factory = AlphaFactory(llm_client)
result = await factory.discover(
    context="Low win rate due to false breakouts",
    ohlcv_data=df,
    forward_returns=returns_series,
)
print(f"Discovered {result.success_count} new factors")
```

Discovered factors are automatically:
1. Validated via AST security checks
2. Executed in sandboxed subprocess
3. Statistically validated (IC, IR, monthly win-rate)
4. Stored in SQLite factor library
5. Registered into IndicatorRegistry

## Using Custom Indicators in Strategies

Once registered, use them in YAML just like built-in indicators:

```yaml
entry:
  triggers:
    - indicator: my_custom_indicator
      op: ">"
      value: 0.05
```

## Next Steps

- [05 — Evolution Guide](05_evolution_guide.md)
- [06 — Real Data Walkthrough](06_real_data_walkthrough.md)
