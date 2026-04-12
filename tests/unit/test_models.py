"""Tests for Pydantic data models."""

from datetime import date

from alphaevo.models.enums import (
    ChangeType,
    ExitReason,
    MarketRegime,
    MarketType,
    SignalDirection,
    StrategyCategory,
)
from alphaevo.models.execution import (
    AntiFitMetrics,
    BenchmarkComparison,
    EvaluationReport,
    OverallMetrics,
    RegimeMetrics,
    StrategyChange,
    TradeSignal,
)
from alphaevo.models.market import (
    EventContextRecord,
    EventContextSeries,
    MarketSnapshot,
    PriceData,
    VolumeData,
)
from alphaevo.models.strategy import (
    MarketHypothesisAssessment,
    MarketHypothesisSummary,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    TakeProfitConfig,
)


class TestMarketModels:
    def test_market_snapshot_minimal(self) -> None:
        snap = MarketSnapshot(
            symbol="000001.SZ",
            date=date(2026, 1, 1),
            market=MarketType.A_SHARE,
            price=PriceData(open=10.0, high=10.5, low=9.8, close=10.2),
            volume=VolumeData(volume=1_000_000),
        )
        assert snap.symbol == "000001.SZ"
        assert snap.market == MarketType.A_SHARE
        assert snap.price.close == 10.2

    def test_event_context_series(self) -> None:
        series = EventContextSeries(
            symbol="AAPL",
            source="mock_feed",
            records=[
                EventContextRecord(
                    date=date(2026, 1, 2),
                    news_sentiment_score=0.72,
                    days_since_event=1,
                )
            ],
        )
        assert series.symbol == "AAPL"
        assert series.source == "mock_feed"
        assert series.records[0].news_sentiment_score == 0.72


class TestStrategyModels:
    def test_strategy_meta_family_id(self) -> None:
        meta = StrategyMeta(
            id="trend_pullback_v3",
            name="test",
            version=3,
            category=StrategyCategory.TREND,
        )
        assert meta.family_id == "trend_pullback"

    def test_strategy_meta_no_version_suffix(self) -> None:
        meta = StrategyMeta(
            id="my_custom_strategy",
            name="test",
            category=StrategyCategory.EVENT,
        )
        assert meta.family_id == "my_custom_strategy"

    def test_strategy_meta_experimental_default(self) -> None:
        meta = StrategyMeta(id="test_v1", name="test")
        assert meta.experimental is False

    def test_strategy_meta_experimental_true(self) -> None:
        meta = StrategyMeta(id="test_v1", name="test", experimental=True)
        assert meta.experimental is True

    def test_strategy_complexity_score(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(id="test_v1", name="test", category=StrategyCategory.TREND),
            description="test",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="rsi", op=">", value=30),
                    StrategyCondition(indicator="ma5", op=">", value=10),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
        )
        assert 0.0 <= strategy.complexity_score <= 1.0

    def test_build_market_hypothesis(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(
                id="trend_pullback_v1",
                name="trend",
                category=StrategyCategory.TREND,
                preferred_regime=["trending_up"],
            ),
            description="Buy orderly pullbacks in strong trends.",
            entry=StrategyEntry(
                logic="and",
                conditions=[
                    StrategyCondition(indicator="relative_strength_20d", op=">", value=0.1),
                    StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.5),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="atr", multiplier=2.0),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
        )

        hypothesis = strategy.build_market_hypothesis()

        assert isinstance(hypothesis, MarketHypothesisSummary)
        assert hypothesis.thesis.startswith("Buy orderly pullbacks")
        assert hypothesis.expected_regimes == ["trending_up"]
        assert "relative_strength_20d" in hypothesis.key_indicators
        assert "High-conviction gating" in hypothesis.signal_style

    def test_assess_market_hypothesis_small_sample(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(id="test_v1", name="test", category=StrategyCategory.REVERSAL),
            description="Fade short-term oversold moves.",
            entry=StrategyEntry(
                conditions=[StrategyCondition(indicator="rsi_14", op="<", value=30)],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
        )
        report = EvaluationReport(
            strategy_id="test_v1",
            overall=OverallMetrics(signal_count=12, win_rate=0.55, avg_return=0.01),
        )

        assessment = strategy.assess_market_hypothesis(report)

        assert isinstance(assessment, MarketHypothesisAssessment)
        assert assessment.status == "unproven_small_sample"
        assert "below the minimum evidence threshold" in assessment.rationale

    def test_assess_market_hypothesis_prefers_thesis_when_regime_break_is_obvious(self) -> None:
        strategy = Strategy(
            meta=StrategyMeta(
                id="trend_v1",
                name="trend",
                category=StrategyCategory.TREND,
                preferred_regime=["trending_up"],
            ),
            description="Buy orderly pullbacks in strong uptrends.",
            entry=StrategyEntry(
                conditions=[
                    StrategyCondition(indicator="relative_strength_20d", op=">", value=0.08),
                    StrategyCondition(indicator="close_above_ma20", op="==", value=True),
                ],
            ),
            exit=StrategyExit(
                stop_loss=StopLossConfig(type="pct", value=0.04),
                take_profit=TakeProfitConfig(type="rr", value=2.0),
            ),
        )
        report = EvaluationReport(
            strategy_id="trend_v1",
            overall=OverallMetrics(signal_count=84, win_rate=0.40, avg_return=0.002),
            anti_overfit=AntiFitMetrics(train_val_gap=0.18),
            benchmark=BenchmarkComparison(excess_return=-0.08),
            by_regime=[
                RegimeMetrics(
                    regime=MarketRegime.TRENDING_UP,
                    win_rate=0.36,
                    avg_return=-0.01,
                    signal_count=60,
                )
            ],
        )

        assessment = strategy.assess_market_hypothesis(report)

        assert assessment.status == "thesis_misaligned"
        assert "overfit symptoms" in assessment.rationale


class TestExecutionModels:
    def test_trade_signal(self) -> None:
        signal = TradeSignal(
            symbol="600519.SH",
            signal_date=date(2026, 1, 1),
            direction=SignalDirection.LONG,
            entry_price=100.0,
            exit_price=108.0,
            exit_reason=ExitReason.TAKE_PROFIT,
            return_pct=0.08,
            holding_days=5,
        )
        assert signal.return_pct == 0.08

    def test_anti_fit_overfit_detection(self) -> None:
        metrics = AntiFitMetrics(
            train_win_rate=0.75,
            val_win_rate=0.55,
            test_win_rate=0.50,
            train_val_gap=0.20,
            val_test_gap=0.05,
        )
        assert metrics.is_overfit is True  # train_val_gap > 0.15

    def test_anti_fit_no_overfit(self) -> None:
        metrics = AntiFitMetrics(
            train_win_rate=0.60,
            val_win_rate=0.58,
            test_win_rate=0.56,
            train_val_gap=0.02,
            val_test_gap=0.02,
        )
        assert metrics.is_overfit is False

    def test_strategy_change(self) -> None:
        change = StrategyChange(
            change_type=ChangeType.TIGHTEN_FILTER,
            target="entry.conditions[0].value",
            from_value=0.08,
            to_value=0.12,
            reason="weak trend names caused most false positives",
        )
        assert change.change_type == ChangeType.TIGHTEN_FILTER

    def test_evaluation_report(self) -> None:
        report = EvaluationReport(
            strategy_id="test_v1",
            overall=OverallMetrics(win_rate=0.61, avg_return=0.034),
        )
        assert report.overall.win_rate == 0.61
