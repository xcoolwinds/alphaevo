"""Tests for StrategyGenerator — mocked LLM generation."""

import pytest

from alphaevo.core.config import LLMConfig
from alphaevo.core.llm import LLMClient
from alphaevo.strategy.generator import StrategyGenerator

_VALID_YAML = """meta:
  id: rsi_mean_reversion_v1
  name: RSI Mean Reversion
  version: 1
  market: us
  category: reversal
  tags: [RSI, mean_reversion]

description: |
  Buy when RSI drops below 30 indicating oversold conditions.

universe:
  market: [us]

entry:
  conditions:
    - indicator: rsi_14
      op: "<"
      value: 30
    - indicator: volume_ratio_1d_5d
      op: ">"
      value: 1.2

exit:
  stop_loss:
    type: pct
    value: 0.03
  take_profit:
    type: rr
    value: 2.0
  max_holding_days: 5

params:
  tunable:
    - target: entry.conditions[indicator=rsi_14].value
      range: [20, 40]
      step: 5
"""


def _mock_llm_chat(llm, response_text):
    class FakeMessage:
        content = response_text

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]

    class FakeLitellm:
        @staticmethod
        def completion(**kwargs):
            return FakeResponse()

    llm._litellm = FakeLitellm()


@pytest.fixture
def llm():
    config = LLMConfig(model="test-model")
    return LLMClient(config)


class TestStrategyGenerator:
    def test_generate_valid_strategy(self, llm):
        _mock_llm_chat(llm, _VALID_YAML)
        gen = StrategyGenerator(llm)
        strategy = gen.generate("RSI mean reversion for US stocks", market="us")

        assert strategy.meta.id == "rsi_mean_reversion_v1"
        assert strategy.meta.name == "RSI Mean Reversion"
        assert len(strategy.entry.conditions) == 2
        assert strategy.exit.stop_loss.value == 0.03

    def test_generate_strips_markdown_fences(self, llm):
        fenced = f"```yaml\n{_VALID_YAML}\n```"
        _mock_llm_chat(llm, fenced)
        gen = StrategyGenerator(llm)
        strategy = gen.generate("RSI reversion")
        assert strategy.meta.id == "rsi_mean_reversion_v1"

    def test_generate_retries_on_parse_error(self, llm):
        """First call returns bad YAML, second returns valid."""
        call_count = [0]

        class FakeMessage:
            def __init__(self, text):
                self.content = text

        class FakeChoice:
            def __init__(self, text):
                self.message = FakeMessage(text)

        class FakeResponse:
            def __init__(self, text):
                self.choices = [FakeChoice(text)]

        class FakeLitellm:
            @staticmethod
            def completion(**kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    return FakeResponse("invalid: yaml: [broken")
                return FakeResponse(_VALID_YAML)

        llm._litellm = FakeLitellm()
        gen = StrategyGenerator(llm)
        strategy = gen.generate("RSI reversion", max_retries=2)
        assert strategy.meta.id == "rsi_mean_reversion_v1"
        assert call_count[0] >= 2  # At least one retry

    def test_generate_fails_after_max_retries(self, llm):
        _mock_llm_chat(llm, "completely invalid garbage")
        gen = StrategyGenerator(llm)
        with pytest.raises(ValueError, match="Failed to generate"):
            gen.generate("something", max_retries=1)

    def test_generate_normalizes_common_yaml_shape_errors(self, llm):
        malformed_but_repairable = """meta:
  id: simple_bullish_momentum_v1
  version: 1
  market: us
  category: trend
  preferred_regime: bull
description: |
  Momentum strategy.
universe:
  market: us
  filters:
    - sector:
        exclude: [Financials]
entry:
  conditions:
    - indicator: rsi_14
      op: "<"
      value: 30
  filters:
    indicator: has_stop_signal
    op: "=="
    value: false
exit:
  stop_loss:
    type: pct
    value: 0.03
  take_profit:
    type: rr
    value: 2.0
params:
  tunable:
    target: entry.conditions[indicator=rsi_14].value
    range: [20, 40]
    step: 5
"""
        _mock_llm_chat(llm, malformed_but_repairable)
        gen = StrategyGenerator(llm)

        strategy = gen.generate("Simple bullish momentum strategy", market="us", max_retries=0)

        assert strategy.meta.id == "simple_bullish_momentum_v1"
        assert strategy.meta.name == "Simple Bullish Momentum"
        assert strategy.meta.preferred_regime == ["bull"]
        assert strategy.universe.market == ["us"]
        assert strategy.universe.filters == []
        assert strategy.entry.filters[0].indicator == "has_stop_signal"


class TestCleanYaml:
    def test_no_fences(self):
        text = "key: value"
        assert StrategyGenerator._clean_yaml(text) == "key: value"

    def test_yaml_fence(self):
        text = "```yaml\nkey: value\n```"
        assert StrategyGenerator._clean_yaml(text) == "key: value"

    def test_bare_fence(self):
        text = "```\nkey: value\n```"
        assert StrategyGenerator._clean_yaml(text) == "key: value"

    def test_whitespace_stripped(self):
        text = "  \n  key: value  \n  "
        assert StrategyGenerator._clean_yaml(text) == "key: value"
