"""Strategy generator — create new strategies from natural language via LLM.

Takes a user description (e.g., "RSI mean reversion for US tech stocks")
and produces a valid Strategy YAML that parses cleanly.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import yaml

from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.dsl.serializer import StrategySerializer

if TYPE_CHECKING:
    from alphaevo.core.llm import LLMClient
    from alphaevo.models.strategy import Strategy

logger = logging.getLogger(__name__)

_GENERATE_SYSTEM = """You are an expert quantitative strategy designer.
Generate a complete stock trading strategy in YAML DSL format.

OHLCV means: Open, High, Low, Close, Volume.

Available indicators (L1 — OHLCV only, always available):
  Trend / moving-average family:
  - maN_above_maM (bool, e.g. ma5_above_ma10, ma50_above_ma180)
  - maN_ge_maM_or_crossing (bool, e.g. ma5_ge_ma10_or_crossing, ma20_ge_ma50_or_crossing)
  - close_to_maN_pct (float, e.g. close_to_ma10_pct, close_to_ma50_pct)
  - close_above_maN (bool, e.g. close_above_ma20, close_above_ma180)
  - close_below_maN (bool, e.g. close_below_ma10, close_below_ma50)
  - deviation_from_maN_pct (float, e.g. deviation_from_ma20_pct, deviation_from_ma180_pct)
  - maN_slope (float, e.g. ma20_slope, ma50_slope)
  - momentum_Nd (float, e.g. momentum_10d, momentum_20d)
  - days_since_high_Nd (int, e.g. days_since_high_20d, days_since_high_55d)
  - days_since_low_Nd (int, e.g. days_since_low_20d, days_since_low_55d)
  - price_position_52w (float)

  Volume / volatility / oscillator family:
  - volume_ratio_1d_Nd (float, e.g. volume_ratio_1d_5d, volume_ratio_1d_20d)
  - avg_volume_Nd (float, e.g. avg_volume_20d, avg_volume_60d)
  - volatility_Nd (float, e.g. volatility_20d, volatility_60d)
  - atr / atr_N (float, e.g. atr, atr_21)
  - rsi_N (float 0-100, e.g. rsi_14, rsi_21)
  - rsi_N_zscore (float, e.g. rsi_14_zscore, rsi_21_zscore)
  - macd_histogram / macd_histogram_fastN_slowM_signalK (float, e.g. macd_histogram, macd_histogram_fast12_slow26_signal9)
  - macd_cross_bullish / macd_cross_bullish_fastN_slowM_signalK (bool)
  - bollinger_band_width / bollinger_band_width_Nd / bollinger_band_width_Nd_stdS (float, e.g. bollinger_band_width, bollinger_band_width_30d, bollinger_band_width_30d_std1p5)
  - price_above_bollinger_upper / price_above_bollinger_upper_Nd / price_above_bollinger_upper_Nd_stdS (bool)
  - price_below_bollinger_lower / price_below_bollinger_lower_Nd / price_below_bollinger_lower_Nd_stdS (bool)

  Pattern / structure family:
  - has_stop_signal (bool)
  - volume_shrink_then_rise (bool)
  - consecutive_up_days (int)
  - consecutive_down_days (int)
  - gap_up_pct (float)
  - body_to_range_ratio (float)

Available indicators (L2 — may degrade in MVP):
  relative_strength_Nd (float, e.g. relative_strength_20d, relative_strength_60d), st_flag (bool),
  sector_heat_rank (int), sector_heat_rising_days (int),
  intra_sector_strength_rank_pct (float)

Condition operators: ==, !=, >, >=, <, <=
Stop loss types: pct, atr, pct_from_low, composite
Take profit types: rr, pct, trailing, target_ma
target_ma options: any maN (e.g. ma5, ma20, ma50, ma180)
Market types: a_share, us, hk
Categories: trend, reversal, event, rotation

Rules:
- Use 3-6 entry conditions (avoid overfitting)
- Prefer currently available indicators instead of inventing new ones
- For supported indicator families, any positive integer window is allowed when the name follows the documented templates (maN/maM, atr_N, rsi_N, macd_histogram_fastN_slowM_signalK, macd_cross_bullish_fastN_slowM_signalK, bollinger_band_width_Nd, bollinger_band_width_Nd_stdS, price_above_bollinger_upper_Nd, price_above_bollinger_upper_Nd_stdS, price_below_bollinger_lower_Nd, price_below_bollinger_lower_Nd_stdS, volume_ratio_1d_Nd, momentum_Nd, avg_volume_Nd, days_since_high_Nd, days_since_low_Nd, volatility_Nd, relative_strength_Nd)
- Always include stop_loss and take_profit
- Prefer entry.triggers for the actual buy event and entry.guards for hard filters. For backward compatibility, entry.conditions/entry.filters are also valid.
- Use exit.triggers when the user describes an explicit sell/exit signal (for example close_below_ma10 or rsi_14 > 75). exit.triggers uses the same list item schema as entry.conditions.
- Include at least 2 tunable parameters
- If a strategy depends heavily on indicator windows, you may tune the period itself via targets like entry.triggers[indicator=rsi_14].indicator, entry.guards[indicator=relative_strength_20d].indicator, entry.conditions[indicator=close_above_ma60].indicator, entry.conditions[indicator=ma5_ge_ma10_or_crossing].indicator.fast, entry.triggers[indicator=macd_histogram].indicator.signal, entry.guards[indicator=bollinger_band_width].indicator.std, or exit.take_profit.target
- preferred_regime should match the strategy style
- meta.name is required
- meta.preferred_regime must be a YAML list, e.g. [trending_up]
- universe.filters must be a YAML list of objects with exactly: field, op, value
- entry.triggers, entry.guards, entry.conditions, and entry.filters must be YAML lists of objects with exactly: indicator, op, value
- params.tunable[*].target must keep the full selector path, e.g. entry.triggers[indicator=rsi_14].value
- If an optional section is hard to express correctly, simplify or omit it instead of inventing a new schema
- Description must explain the logic clearly
"""

_GENERATE_USER = """Create a strategy based on this description:

"{description}"

Respond with ONLY the YAML content (no markdown fences, no explanation).
The YAML must have these sections: meta, description, universe, entry, exit, params.

meta.id should be a snake_case name ending with _v1.
meta.name must be present.
meta.market should be "{market}".
"""

_REFINE_USER = """The generated YAML had parsing errors:
{errors}

Original YAML:
```yaml
{yaml_content}
```

Fix the YAML so it parses correctly. Return ONLY the corrected YAML."""


class StrategyGenerator:
    """Generate new strategies from natural language descriptions."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self._parser = StrategyParser()
        self._serializer = StrategySerializer()

    def generate(
        self,
        description: str,
        market: str = "a_share",
        max_retries: int = 3,
    ) -> Strategy:
        """Generate a strategy from a natural language description.

        Makes up to max_retries attempts to fix parsing errors.
        Raises ValueError if all attempts fail.
        """
        messages = [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {
                "role": "user",
                "content": _GENERATE_USER.format(
                    description=description,
                    market=market,
                ),
            },
        ]

        yaml_content = self.llm.chat(
            messages,
            temperature=0.2,
            max_tokens=4096,
        )
        yaml_content = self._clean_yaml(yaml_content)

        # Try to parse, retry on failure
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            normalized = self._normalize_yaml_candidate(yaml_content)
            candidates = [yaml_content]
            if normalized is not None and normalized != yaml_content:
                candidates.append(normalized)

            parse_errors: list[str] = []
            for candidate in candidates:
                try:
                    strategy = self._parse_and_validate(candidate)
                    logger.info(
                        "Strategy generated: %s (attempt %d)",
                        strategy.meta.id,
                        attempt + 1,
                    )
                    return strategy
                except Exception as e:
                    last_error = e
                    parse_errors.append(str(e))

            if attempt < max_retries:
                logger.warning(
                    "Parse failed (attempt %d): %s — retrying",
                    attempt + 1,
                    last_error,
                )
                error_text = (
                    "\n".join(dict.fromkeys(parse_errors)) if parse_errors else str(last_error)
                )
                yaml_content = self._refine(normalized or yaml_content, error_text)
            else:
                assert last_error is not None
                raise ValueError(
                    f"Failed to generate valid strategy after {max_retries + 1} attempts: {last_error}"
                ) from last_error

        raise ValueError(
            f"Failed to generate valid strategy after {max_retries + 1} attempts: {last_error}"
        )

    def _refine(self, yaml_content: str, error: str) -> str:
        """Ask LLM to fix parsing errors."""
        messages = [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {
                "role": "user",
                "content": _REFINE_USER.format(
                    errors=error,
                    yaml_content=yaml_content,
                ),
            },
        ]
        result = self.llm.chat(messages, temperature=0.3, max_tokens=4096)
        return self._clean_yaml(result)

    def _parse_and_validate(self, yaml_content: str) -> Strategy:
        """Parse YAML and enforce semantic validation."""
        strategy = self._parser.parse_yaml(yaml_content)
        self._parser.assert_valid(strategy)
        return strategy

    def _normalize_yaml_candidate(self, yaml_content: str) -> str | None:
        """Repair common structured-output mistakes without another LLM call."""
        try:
            raw = yaml.safe_load(yaml_content)
        except yaml.YAMLError:
            return None
        if not isinstance(raw, dict):
            return None

        normalized = self._normalize_raw_strategy(raw)
        dumped = yaml.dump(
            normalized,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )
        return self._clean_yaml(dumped)

    def _normalize_raw_strategy(self, raw: dict[str, Any]) -> dict[str, Any]:
        """Best-effort normalization for frequent YAML shape mistakes."""
        normalized = dict(raw)

        meta = normalized.get("meta")
        if isinstance(meta, dict):
            meta = dict(meta)
            strategy_id = meta.get("id")
            if isinstance(strategy_id, str) and strategy_id and not meta.get("name"):
                family = strategy_id.rsplit("_v", 1)[0]
                meta["name"] = family.replace("_", " ").strip().title()
            preferred_regime = meta.get("preferred_regime")
            if isinstance(preferred_regime, str):
                meta["preferred_regime"] = [preferred_regime]
            tags = meta.get("tags")
            if isinstance(tags, str):
                meta["tags"] = [tags]
            normalized["meta"] = meta

        universe = normalized.get("universe")
        if isinstance(universe, dict):
            universe = dict(universe)
            market = universe.get("market")
            if isinstance(market, str):
                universe["market"] = [market]
            universe["filters"] = self._normalize_object_list(
                universe.get("filters"),
                required_keys=("field", "op", "value"),
            )
            normalized["universe"] = universe

        entry = normalized.get("entry")
        if isinstance(entry, dict):
            entry = dict(entry)
            entry["conditions"] = self._normalize_object_list(
                entry.get("conditions"),
                required_keys=("indicator", "op", "value"),
            )
            entry["filters"] = self._normalize_object_list(
                entry.get("filters"),
                required_keys=("indicator", "op", "value"),
            )
            normalized["entry"] = entry

        if isinstance(normalized.get("exit"), dict):
            exit_block = dict(normalized["exit"])
            stop_loss = exit_block.get("stop_loss")
            if isinstance(stop_loss, dict):
                stop_loss = dict(stop_loss)
                if "conditions" in stop_loss:
                    stop_loss["conditions"] = self._normalize_object_list(
                        stop_loss.get("conditions"),
                        required_keys=("indicator", "op", "value"),
                    )
                exit_block["stop_loss"] = stop_loss
            normalized["exit"] = exit_block

        params = normalized.get("params")
        if isinstance(params, dict):
            params = dict(params)
            tunable = params.get("tunable")
            if isinstance(tunable, dict):
                params["tunable"] = [tunable]
            elif isinstance(tunable, list):
                cleaned_tunable = []
                for item in tunable:
                    if isinstance(item, dict) and {"target", "range", "step"} <= item.keys():
                        cleaned_tunable.append(item)
                params["tunable"] = cleaned_tunable
            normalized["params"] = params

        return normalized

    @staticmethod
    def _normalize_object_list(
        value: Any,
        *,
        required_keys: tuple[str, str, str],
    ) -> list[dict[str, Any]]:
        """Keep only list entries that match the expected object shape."""
        if value is None:
            return []
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list):
            return []

        cleaned: list[dict[str, Any]] = []
        for item in value:
            if isinstance(item, dict) and set(required_keys) <= item.keys():
                cleaned.append(item)
        return cleaned

    @staticmethod
    def _clean_yaml(text: str) -> str:
        """Strip markdown code fences from YAML response."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```yaml) and last line (```)
            start = 1
            end = len(lines)
            for i in range(len(lines) - 1, 0, -1):
                if lines[i].strip() == "```":
                    end = i
                    break
            text = "\n".join(lines[start:end])
        meta_idx = text.find("meta:")
        if meta_idx > 0:
            text = text[meta_idx:]
        return text
