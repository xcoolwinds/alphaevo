"""Tests for Strategy serializer (Strategy → YAML round-trip)."""

from pathlib import Path

import pytest
import yaml

from alphaevo.models.enums import MarketType, StrategyCategory
from alphaevo.models.strategy import (
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
)
from alphaevo.strategy.dsl.parser import StrategyParser
from alphaevo.strategy.dsl.serializer import StrategySerializer

BUILTIN_DIR = Path(__file__).parent.parent.parent / "strategies" / "builtin"


class TestStrategySerializer:
    """Test StrategySerializer."""

    def setup_method(self) -> None:
        self.serializer = StrategySerializer()
        self.parser = StrategyParser()

    def _make_simple_strategy(self) -> Strategy:
        return Strategy(
            meta=StrategyMeta(
                id="test_strategy_v1",
                name="Test Strategy",
                version=1,
                market=MarketType.A_SHARE,
                category=StrategyCategory.TREND,
                tags=["test"],
            ),
            description="A simple test strategy for unit testing.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op="<", value=30),
                    StrategyCondition(indicator="ma5_above_ma10", op="==", value=True),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
                max_holding_days=10,
            ),
        )

    def test_to_dict_basic(self) -> None:
        strategy = self._make_simple_strategy()
        d = self.serializer.to_dict(strategy)

        assert d["meta"]["id"] == "test_strategy_v1"
        assert d["meta"]["name"] == "Test Strategy"
        assert d["meta"]["market"] == "a_share"
        assert d["meta"]["category"] == "trend"
        assert "family_id" not in d["meta"]  # computed field excluded
        assert len(d["entry"]["conditions"]) == 2
        assert d["exit"]["stop_loss"]["type"] == "pct"

    def test_to_yaml_valid(self) -> None:
        strategy = self._make_simple_strategy()
        yaml_str = self.serializer.to_yaml(strategy)

        # Should be valid YAML
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)
        assert parsed["meta"]["id"] == "test_strategy_v1"

    def test_to_file(self, tmp_path: Path) -> None:
        strategy = self._make_simple_strategy()
        out_path = tmp_path / "output" / "test.yaml"
        self.serializer.to_file(strategy, out_path)

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        parsed = yaml.safe_load(content)
        assert parsed["meta"]["id"] == "test_strategy_v1"

    def test_roundtrip_simple(self) -> None:
        """Strategy → YAML → Strategy should preserve key fields."""
        original = self._make_simple_strategy()
        yaml_str = self.serializer.to_yaml(original)
        restored = self.parser.parse_yaml(yaml_str)

        assert restored.meta.id == original.meta.id
        assert restored.meta.name == original.meta.name
        assert restored.meta.version == original.meta.version
        assert restored.meta.category == original.meta.category
        assert len(restored.entry.conditions) == len(original.entry.conditions)
        assert restored.exit.stop_loss.type == original.exit.stop_loss.type
        assert restored.exit.stop_loss.value == original.exit.stop_loss.value
        assert restored.exit.take_profit.type == original.exit.take_profit.type
        assert restored.exit.max_holding_days == original.exit.max_holding_days

    def test_roundtrip_builtin_strategies(self) -> None:
        """All builtin strategies should survive parse → serialize → parse."""
        if not BUILTIN_DIR.is_dir():
            pytest.skip("Builtin strategies directory not found")

        strategies = self.parser.parse_directory(BUILTIN_DIR)
        assert len(strategies) >= 4

        for original in strategies:
            yaml_str = self.serializer.to_yaml(original)
            restored = self.parser.parse_yaml(yaml_str)

            assert restored.meta.id == original.meta.id, f"ID mismatch for {original.meta.id}"
            assert restored.meta.category == original.meta.category
            assert len(restored.entry.conditions) == len(original.entry.conditions)
            assert restored.exit.max_holding_days == original.exit.max_holding_days

    def test_to_dict_strips_none_values(self) -> None:
        strategy = self._make_simple_strategy()
        d = self.serializer.to_dict(strategy)

        # parent_id is None, should be stripped
        assert "parent_id" not in d.get("meta", {})

    def test_to_dict_strips_empty_collections(self) -> None:
        strategy = self._make_simple_strategy()
        d = self.serializer.to_dict(strategy)

        # Empty filters and params should be stripped
        assert "filters" not in d.get("entry", {})

    def test_tunable_params_roundtrip(self) -> None:
        """Tunable params should survive round-trip."""
        strategy = Strategy(
            meta=StrategyMeta(
                id="param_test_v1",
                name="Param Test",
                version=1,
                category=StrategyCategory.REVERSAL,
            ),
            description="Test tunable params.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op="<", value=30),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
            params=StrategyParams(
                tunable=[
                    TunableParam(
                        target="entry.conditions[indicator=rsi_14].value",
                        range=(20, 40),
                        step=1.0,
                        label="RSI threshold",
                    ),
                ]
            ),
        )
        yaml_str = self.serializer.to_yaml(strategy)
        restored = self.parser.parse_yaml(yaml_str)

        assert len(restored.params.tunable) == 1
        assert restored.params.tunable[0].target == "entry.conditions[indicator=rsi_14].value"
        assert restored.params.tunable[0].range == (20.0, 40.0)
        assert restored.params.tunable[0].step == 1.0

    def test_ma_period_tunable_roundtrip(self) -> None:
        """MA-period tunables should survive round-trip."""
        strategy = Strategy(
            meta=StrategyMeta(
                id="ma_period_test_v1",
                name="MA Period Test",
                version=1,
                category=StrategyCategory.TREND,
            ),
            description="Test MA period tunables.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="close_above_ma60", op="==", value=True),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="target_ma", target="ma60"),
            ),
            params=StrategyParams(
                tunable=[
                    TunableParam(
                        target="entry.conditions[indicator=close_above_ma60].indicator",
                        range=(40, 120),
                        step=5.0,
                        label="Trend MA period",
                    ),
                    TunableParam(
                        target="exit.take_profit.target",
                        range=(20, 80),
                        step=5.0,
                        label="Exit MA period",
                    ),
                ]
            ),
        )
        yaml_str = self.serializer.to_yaml(strategy)
        restored = self.parser.parse_yaml(yaml_str)

        assert [param.target for param in restored.params.tunable] == [
            "entry.conditions[indicator=close_above_ma60].indicator",
            "exit.take_profit.target",
        ]

    def test_atr_stop_loss_period_roundtrip(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(
                id="atr_stop_test_v1",
                name="ATR Stop Test",
                version=1,
                category=StrategyCategory.TREND,
            ),
            description="Test ATR stop period round-trip.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi_14", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="atr", multiplier=2.0, atr_period=21),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
            params=StrategyParams(
                tunable=[
                    TunableParam(
                        target="exit.stop_loss.atr_period",
                        range=(7, 30),
                        step=1.0,
                        label="ATR stop period",
                    ),
                ]
            ),
        )

        yaml_str = self.serializer.to_yaml(strategy)
        restored = self.parser.parse_yaml(yaml_str)

        assert restored.exit.stop_loss.type == "atr"
        assert restored.exit.stop_loss.multiplier == 2.0
        assert restored.exit.stop_loss.atr_period == 21
        assert restored.params.tunable[0].target == "exit.stop_loss.atr_period"

    def test_macd_component_tunables_roundtrip(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(
                id="macd_tunable_test_v1",
                name="MACD Tunable Test",
                version=1,
                category=StrategyCategory.TREND,
            ),
            description="Test MACD component tunables.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="macd_histogram", op=">", value=0),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
            params=StrategyParams(
                tunable=[
                    TunableParam(
                        target="entry.conditions[indicator=macd_histogram].indicator.fast",
                        range=(6, 18),
                        step=1.0,
                    ),
                    TunableParam(
                        target="entry.conditions[indicator=macd_histogram].indicator.slow",
                        range=(20, 40),
                        step=1.0,
                    ),
                    TunableParam(
                        target="entry.conditions[indicator=macd_histogram].indicator.signal",
                        range=(5, 15),
                        step=1.0,
                    ),
                ]
            ),
        )
        yaml_str = self.serializer.to_yaml(strategy)
        restored = self.parser.parse_yaml(yaml_str)

        assert [param.target for param in restored.params.tunable] == [
            "entry.conditions[indicator=macd_histogram].indicator.fast",
            "entry.conditions[indicator=macd_histogram].indicator.slow",
            "entry.conditions[indicator=macd_histogram].indicator.signal",
        ]

    def test_bollinger_std_tunable_roundtrip(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(
                id="bollinger_std_test_v1",
                name="Bollinger Std Test",
                version=1,
                category=StrategyCategory.TREND,
            ),
            description="Test Bollinger std tunable.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="bollinger_band_width", op="<", value=0.2),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
            params=StrategyParams(
                tunable=[
                    TunableParam(
                        target="entry.conditions[indicator=bollinger_band_width].indicator.std",
                        range=(1.0, 3.0),
                        step=0.5,
                    ),
                ]
            ),
        )
        yaml_str = self.serializer.to_yaml(strategy)
        restored = self.parser.parse_yaml(yaml_str)

        assert restored.params.tunable[0].target == (
            "entry.conditions[indicator=bollinger_band_width].indicator.std"
        )
