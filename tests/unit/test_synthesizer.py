"""Tests for FactorSynthesizer (LLM-driven factor generation)."""

from __future__ import annotations

import pytest

from alphaevo.alpha_factory.synthesizer import FactorHypothesis, FactorSynthesizer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockLLM:
    """Minimal mock for LLMClient.chat_json()."""

    def __init__(self, response=None, raise_error=False):
        self._response = response
        self._raise = raise_error
        self.calls: list[dict] = []

    def chat_json(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        if self._raise:
            raise RuntimeError("LLM unavailable")
        return self._response


_VALID_FACTOR = {
    "name": "vol_ratio_5_20",
    "description": "Volume 5/20 ratio",
    "rationale": "Detects volume expansion",
    "code": "def compute(df, idx):\n    return 1.0",
    "expected_direction": "positive",
}


# ---------------------------------------------------------------------------
# Tests — _parse_response
# ---------------------------------------------------------------------------


class TestParseResponse:
    def test_parse_list(self):
        raw = [_VALID_FACTOR]
        results = FactorSynthesizer._parse_response(raw)
        assert len(results) == 1
        assert results[0].name == "vol_ratio_5_20"

    def test_parse_dict_wrapped(self):
        raw = {"factors": [_VALID_FACTOR]}
        results = FactorSynthesizer._parse_response(raw)
        assert len(results) == 1

    def test_parse_single_dict(self):
        raw = dict(_VALID_FACTOR)
        results = FactorSynthesizer._parse_response(raw)
        assert len(results) == 1

    def test_parse_empty_list(self):
        assert FactorSynthesizer._parse_response([]) == []

    def test_parse_non_list(self):
        assert FactorSynthesizer._parse_response("bad input") == []

    def test_strips_code_fences(self):
        factor = dict(_VALID_FACTOR)
        factor["code"] = "```python\ndef compute(df, idx):\n    return 1.0\n```"
        results = FactorSynthesizer._parse_response([factor])
        assert "```" not in results[0].code
        assert results[0].code.startswith("def compute")

    def test_skips_invalid_items(self):
        raw = [_VALID_FACTOR, {"invalid": True}, {"name": "x"}]
        results = FactorSynthesizer._parse_response(raw)
        # Only the first valid factor survives
        assert len(results) == 1

    def test_candidates_wrapper_key(self):
        raw = {"candidates": [_VALID_FACTOR]}
        results = FactorSynthesizer._parse_response(raw)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Tests — generate
# ---------------------------------------------------------------------------


class TestGenerate:
    @pytest.mark.asyncio
    async def test_generate_success(self):
        llm = _MockLLM(response=[_VALID_FACTOR])
        synth = FactorSynthesizer(llm)
        results = await synth.generate(
            context="Low win rate",
            existing_factors=["rsi_14"],
        )
        assert len(results) == 1
        assert results[0].name == "vol_ratio_5_20"
        # Verify LLM was called
        assert len(llm.calls) == 1

    @pytest.mark.asyncio
    async def test_generate_llm_error(self):
        llm = _MockLLM(raise_error=True)
        synth = FactorSynthesizer(llm)
        results = await synth.generate(
            context="error case",
            existing_factors=[],
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_generate_empty_existing(self):
        llm = _MockLLM(response=[_VALID_FACTOR])
        synth = FactorSynthesizer(llm)
        await synth.generate(context="test", existing_factors=[])
        msg = llm.calls[0]["messages"][1]["content"]
        assert "none" in msg

    @pytest.mark.asyncio
    async def test_generate_with_failed_attempts(self):
        llm = _MockLLM(response=[_VALID_FACTOR])
        synth = FactorSynthesizer(llm)
        await synth.generate(
            context="test",
            existing_factors=["ma5"],
            failed_attempts=["bad_factor", "worse_factor"],
        )
        msg = llm.calls[0]["messages"][1]["content"]
        assert "bad_factor" in msg


# ---------------------------------------------------------------------------
# Tests — refine
# ---------------------------------------------------------------------------


class TestRefine:
    @pytest.mark.asyncio
    async def test_refine_success(self):
        llm = _MockLLM(response=[_VALID_FACTOR])
        synth = FactorSynthesizer(llm)
        hyp = FactorHypothesis(**_VALID_FACTOR)
        result = await synth.refine(hyp, error_message="division by zero")
        assert result is not None
        assert result.name == "vol_ratio_5_20"

    @pytest.mark.asyncio
    async def test_refine_llm_error(self):
        llm = _MockLLM(raise_error=True)
        synth = FactorSynthesizer(llm)
        hyp = FactorHypothesis(**_VALID_FACTOR)
        result = await synth.refine(hyp, error_message="error")
        assert result is None


# ---------------------------------------------------------------------------
# Tests — FactorHypothesis model
# ---------------------------------------------------------------------------


class TestFactorHypothesis:
    def test_default_direction(self):
        hyp = FactorHypothesis(
            name="test",
            description="d",
            rationale="r",
            code="def compute(df, idx): return 0.0",
        )
        assert hyp.expected_direction == "positive"

    def test_negative_direction(self):
        hyp = FactorHypothesis(
            name="test",
            description="d",
            rationale="r",
            code="c",
            expected_direction="negative",
        )
        assert hyp.expected_direction == "negative"
