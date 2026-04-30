"""Deterministic strategy drafting from natural-language ideas.

The builder is intentionally rule based. It gives users a no-LLM path from a
plain strategy idea to an executable DSL, then lets the existing backtest and
evolution pipeline do the statistical work.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Literal

from alphaevo.models.enums import MarketType, StrategyCategory, StrategyStatus
from alphaevo.models.strategy import (
    ExecutionConfig,
    MarketRuleConfig,
    StopLossConfig,
    Strategy,
    StrategyCondition,
    StrategyEntry,
    StrategyExit,
    StrategyMeta,
    StrategyParams,
    TakeProfitConfig,
    TunableParam,
    UniverseConfig,
)

ConditionOp = Literal["==", "!=", ">", ">=", "<", "<="]


class StrategyDraftBuilder:
    """Build and revise executable strategy DSLs from concise user intent."""

    def from_text(
        self,
        description: str,
        *,
        market: str | MarketType = MarketType.A_SHARE,
        strategy_id: str | None = None,
        name: str | None = None,
    ) -> Strategy:
        """Create a valid long-only strategy draft from natural-language text."""
        text = _normalize_text(description)
        if not text:
            raise ValueError("Strategy description is required")
        if _requests_short_strategy(text):
            raise ValueError(
                "Short-selling/open-short strategies are not supported by the "
                "current backtest engine. Express bearish ideas as long exit "
                "rules for now."
            )

        market_type = _parse_market(market)
        category = _infer_category(text)
        style = _infer_style(text, category)
        triggers = self._entry_conditions(text, category, style)
        guards = self._entry_filters(text, market_type, style)
        exit_rules = self._exit_rules(text)

        meta = StrategyMeta(
            id=strategy_id or _make_strategy_id(text, category, style),
            name=name or _make_strategy_name(category, style),
            market=market_type,
            category=category,
            tags=_tags_for(category, style, text),
            status=StrategyStatus.DRAFT,
            preferred_regime=_preferred_regimes(category, style),
            experimental=category in {StrategyCategory.EVENT, StrategyCategory.ROTATION},
        )
        entry = StrategyEntry(
            logic="and",
            triggers=triggers,
            guards=guards,
            execution=ExecutionConfig(
                timing="breakout_high" if _contains_any(text, _CONFIRMATION_TERMS) else "next_open"
            ),
        )
        return Strategy(
            meta=meta,
            description=description.strip(),
            universe=_universe_for(market_type),
            entry=entry,
            exit=exit_rules,
            params=StrategyParams(
                tunable=self._build_tunables(triggers, exit_rules, guards=guards)
            ),
            market_rules=_market_rules_for(market_type),
        )

    def revise(
        self,
        strategy: Strategy,
        instruction: str,
        *,
        strategy_id: str | None = None,
        name: str | None = None,
    ) -> Strategy:
        """Apply a bounded, explainable revision to an existing strategy."""
        text = _normalize_text(instruction)
        if not text:
            raise ValueError("Revision instruction is required")
        if _requests_short_strategy(text):
            raise ValueError(
                "Short-selling/open-short revisions are not supported by the "
                "current backtest engine."
            )

        revised = strategy.model_copy(deep=True)
        revised.meta.id = strategy_id or _next_version_id(strategy.meta.id)
        revised.meta.parent_id = strategy.meta.id
        revised.meta.version = strategy.meta.version + 1
        revised.meta.name = name or f"{strategy.meta.name} Revised"
        revised.meta.status = StrategyStatus.DRAFT
        revised.meta.created_at = datetime.now(timezone.utc)
        revised.description = _append_revision_note(strategy.description, instruction)
        trigger_bucket_name = "triggers" if revised.entry.triggers else "conditions"
        guard_bucket_name = "guards" if revised.entry.guards else "filters"
        trigger_bucket = getattr(revised.entry, trigger_bucket_name)
        guard_bucket = getattr(revised.entry, guard_bucket_name)

        if _contains_any(text, _TIGHTEN_TERMS):
            _tighten_conditions(trigger_bucket)
            _tighten_conditions(guard_bucket)
            _scale_stop_loss(revised.exit.stop_loss, factor=0.85, floor=0.015)
            _ensure_filter(guard_bucket, "volatility_20d", "<", 0.06)

        if _contains_any(text, _LOOSEN_TERMS):
            _loosen_conditions(trigger_bucket)
            _loosen_conditions(guard_bucket)
            _scale_stop_loss(revised.exit.stop_loss, factor=1.15, cap=0.12)

        if _contains_any(text, _CONFIRMATION_TERMS):
            if revised.entry.execution is None:
                revised.entry.execution = ExecutionConfig()
            revised.entry.execution.timing = "breakout_high"
            _ensure_condition(trigger_bucket, "momentum_10d", ">", 0.02)
            _ensure_condition(trigger_bucket, "body_to_range_ratio", ">", 0.45)

        if _contains_any(text, _VOLUME_TERMS):
            _ensure_condition(trigger_bucket, "volume_ratio_1d_5d", ">", 1.3)

        if _contains_any(text, _DRAWDOWN_TERMS):
            _scale_stop_loss(revised.exit.stop_loss, factor=0.8, floor=0.015)
            _ensure_filter(guard_bucket, "volatility_20d", "<", 0.05)

        if _contains_any(text, _PROFIT_RATIO_TERMS):
            _scale_take_profit(revised.exit.take_profit, factor=1.15, cap=5.0)

        new_exit_triggers = _exit_triggers_for(text)
        if new_exit_triggers:
            revised.exit.triggers = _merge_conditions(revised.exit.triggers, new_exit_triggers)

        holding_days = _extract_holding_days(text)
        if holding_days is not None:
            revised.exit.max_holding_days = holding_days

        stop_pct = _extract_percent_after(text, ("止损", "stop loss", "stop-loss"))
        if stop_pct is not None:
            revised.exit.stop_loss.type = "pct"
            revised.exit.stop_loss.value = stop_pct

        take_profit_pct = _extract_percent_after(
            text,
            ("止盈", "take profit", "take-profit"),
        )
        if take_profit_pct is not None:
            revised.exit.take_profit.type = "pct"
            revised.exit.take_profit.value = take_profit_pct

        revised.params.tunable = _merge_tunables(
            revised.params.tunable,
            self._build_tunables(
                trigger_bucket,
                revised.exit,
                guards=guard_bucket,
                trigger_bucket=trigger_bucket_name,
                guard_bucket=guard_bucket_name,
            ),
        )
        return revised

    def _entry_conditions(
        self,
        text: str,
        category: StrategyCategory,
        style: str,
    ) -> list[StrategyCondition]:
        if category == StrategyCategory.ROTATION:
            return [
                StrategyCondition(indicator="sector_heat_rank", op="<=", value=5),
                StrategyCondition(indicator="sector_fund_flow_positive", op="==", value=True),
                StrategyCondition(indicator="intra_sector_strength_rank_pct", op="<=", value=0.30),
                StrategyCondition(indicator="momentum_10d", op=">", value=0.0),
            ]

        if category == StrategyCategory.EVENT:
            return [
                StrategyCondition(indicator="news_sentiment_score", op=">", value=0.55),
                StrategyCondition(indicator="negative_news_score", op="<", value=0.40),
                StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.20),
                StrategyCondition(indicator="price_above_pre_event", op="==", value=True),
            ]

        if category == StrategyCategory.REVERSAL:
            rsi_value = 30 if _contains_any(text, _STRICT_TERMS) else 35
            return [
                StrategyCondition(indicator="rsi_14", op="<", value=rsi_value),
                StrategyCondition(indicator="deviation_from_ma20_pct", op="<", value=-0.05),
                StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=0.80),
            ]

        if style == "breakout":
            return [
                StrategyCondition(indicator="breakout_high_20d", op="==", value=True),
                StrategyCondition(indicator="volume_ratio_1d_20d", op=">=", value=1.20),
                StrategyCondition(indicator="body_to_range_ratio", op=">=", value=0.45),
            ]

        return [
            StrategyCondition(indicator="ma20_slope", op=">", value=0.0),
            StrategyCondition(indicator="close_to_ma10_pct", op="<=", value=0.02),
            StrategyCondition(indicator="volume_ratio_1d_5d", op=">", value=1.20),
            StrategyCondition(indicator="close_above_ma20", op="==", value=True),
        ]

    def _entry_filters(
        self,
        text: str,
        market: MarketType,
        style: str,
    ) -> list[StrategyCondition]:
        filters: list[StrategyCondition] = []
        if style == "breakout":
            filters.extend(
                [
                    StrategyCondition(indicator="close_above_ma20", op="==", value=True),
                    StrategyCondition(indicator="ma20_slope", op=">", value=0.0),
                    StrategyCondition(indicator="price_position_120d", op=">=", value=0.65),
                ]
            )
            if _contains_any(text, _COMPRESSION_TERMS):
                filters.append(
                    StrategyCondition(indicator="bollinger_band_width_20d", op="<=", value=0.14)
                )
        if market == MarketType.A_SHARE or _contains_any(text, ("st", "退市", "风险警示")):
            filters.append(StrategyCondition(indicator="st_flag", op="==", value=False))
        if _contains_any(text, _DRAWDOWN_TERMS):
            filters.append(StrategyCondition(indicator="volatility_20d", op="<", value=0.06))
        if _contains_any(text, ("停牌", "跌停", "异常")):
            filters.append(StrategyCondition(indicator="has_stop_signal", op="==", value=False))
        return filters

    def _exit_rules(self, text: str) -> StrategyExit:
        stop_value = _extract_percent_after(text, ("止损", "stop loss", "stop-loss"))
        if stop_value is None:
            if _contains_any(text, _DRAWDOWN_TERMS | _STRICT_TERMS):
                stop_value = 0.03
            elif _contains_any(text, _LOOSEN_TERMS):
                stop_value = 0.06
            else:
                stop_value = 0.04

        take_profit_pct = _extract_percent_after(text, ("止盈", "take profit", "take-profit"))
        if take_profit_pct is not None:
            take_profit = TakeProfitConfig(type="pct", value=take_profit_pct)
        elif _contains_any(text, _BREAKOUT_TERMS):
            take_profit = TakeProfitConfig(type="trailing", trigger_pct=0.08, trail_pct=0.04)
        else:
            take_profit = TakeProfitConfig(
                type="rr",
                value=2.5 if _contains_any(text, _PROFIT_RATIO_TERMS) else 2.0,
            )

        holding_days = _extract_holding_days(text)
        if holding_days is None:
            holding_days = 5 if _contains_any(text, ("短线", "短期", "swing", "short term")) else 10

        return StrategyExit(
            triggers=_exit_triggers_for(text),
            stop_loss=StopLossConfig(type="pct", value=stop_value),
            take_profit=take_profit,
            max_holding_days=holding_days,
        )

    def _build_tunables(
        self,
        triggers: list[StrategyCondition],
        exit_rules: StrategyExit,
        *,
        guards: list[StrategyCondition] | None = None,
        trigger_bucket: str = "triggers",
        guard_bucket: str = "guards",
    ) -> list[TunableParam]:
        tunables: list[TunableParam] = []
        seen_targets: set[str] = set()
        for bucket_name, conditions in (
            (trigger_bucket, triggers),
            (guard_bucket, guards or []),
        ):
            for condition in conditions:
                target = f"entry.{bucket_name}[indicator={condition.indicator}].value"
                spec = _TUNABLE_SPECS.get(condition.indicator)
                if spec is not None and target not in seen_targets:
                    lo, hi, step, label = spec
                    tunables.append(
                        TunableParam(target=target, range=(lo, hi), step=step, label=label)
                    )
                    seen_targets.add(target)
                period_spec = _PERIOD_TUNABLE_SPECS.get(_period_tunable_key(condition.indicator))
                if period_spec is not None:
                    period_target = (
                        f"entry.{bucket_name}[indicator={condition.indicator}].indicator"
                    )
                    if period_target not in seen_targets:
                        lo, hi, step, label = period_spec
                        tunables.append(
                            TunableParam(
                                target=period_target,
                                range=(lo, hi),
                                step=step,
                                label=label,
                            )
                        )
                        seen_targets.add(period_target)

        if exit_rules.stop_loss.value is not None:
            tunables.append(
                TunableParam(
                    target="exit.stop_loss.value",
                    range=(0.02, 0.08),
                    step=0.005,
                    label="Stop loss",
                )
            )
        if exit_rules.take_profit.value is not None:
            if exit_rules.take_profit.type == "pct":
                tunables.append(
                    TunableParam(
                        target="exit.take_profit.value",
                        range=(0.03, 0.20),
                        step=0.005,
                        label="Take profit pct",
                    )
                )
            else:
                tunables.append(
                    TunableParam(
                        target="exit.take_profit.value",
                        range=(1.2, 3.5),
                        step=0.1,
                        label="Reward/risk ratio",
                    )
                )
        if exit_rules.take_profit.type == "trailing":
            tunables.extend(
                [
                    TunableParam(
                        target="exit.take_profit.trigger_pct",
                        range=(0.04, 0.14),
                        step=0.01,
                        label="Trailing profit trigger",
                    ),
                    TunableParam(
                        target="exit.take_profit.trail_pct",
                        range=(0.02, 0.08),
                        step=0.01,
                        label="Trailing giveback",
                    ),
                ]
            )
        return tunables


_BREAKOUT_TERMS = frozenset(("突破", "新高", "放量突破", "breakout", "52w", "high"))
_COMPRESSION_TERMS = frozenset(
    ("收缩", "缩量整理", "窄幅", "低波动", "平台", "compression", "squeeze", "range contraction")
)
_PULLBACK_TERMS = frozenset(("回踩", "回调", "pullback", "均线附近", "低吸"))
_REVERSAL_TERMS = frozenset(("超跌", "反转", "reversal", "mean reversion", "oversold", "rsi"))
_ROTATION_TERMS = frozenset(("板块", "轮动", "sector", "rotation", "行业"))
_EVENT_TERMS = frozenset(("事件", "公告", "新闻", "业绩", "event", "news", "earnings"))
_STRICT_TERMS = frozenset(("严格", "保守", "高胜率", "精选", "strict", "conservative"))
_TIGHTEN_TERMS = frozenset(
    ("减少交易", "更严格", "严格", "高胜率", "保守", "精选", "tighten", "stricter")
)
_LOOSEN_TERMS = frozenset(("放宽", "更多交易", "提高频率", "aggressive", "loosen", "more trades"))
_CONFIRMATION_TERMS = frozenset(("右侧", "确认", "突破确认", "breakout", "confirmation"))
_VOLUME_TERMS = frozenset(("放量", "成交量", "量能", "volume"))
_DRAWDOWN_TERMS = frozenset(("低回撤", "降低回撤", "控制回撤", "回撤", "drawdown", "risk"))
_PROFIT_RATIO_TERMS = frozenset(("盈亏比", "利润空间", "提高收益", "profit ratio", "reward risk"))
_SHORT_TERMS = frozenset(("做空", "卖空", "开空", "融券", "short sell", "open short"))
_EXIT_SIGNAL_TERMS = frozenset(("卖出", "退出", "跌破", "止盈后", "exit", "sell"))

_TUNABLE_SPECS: dict[str, tuple[float, float, float, str]] = {
    "sector_heat_rank": (3, 15, 1, "Sector heat rank"),
    "intra_sector_strength_rank_pct": (0.10, 0.60, 0.05, "Intra-sector rank"),
    "momentum_10d": (0.00, 0.12, 0.005, "10-day momentum"),
    "news_sentiment_score": (0.45, 0.75, 0.05, "News sentiment"),
    "negative_news_score": (0.20, 0.60, 0.05, "Negative news cap"),
    "rsi_14": (20, 45, 1, "RSI threshold"),
    "deviation_from_ma20_pct": (-0.12, -0.02, 0.005, "MA20 deviation"),
    "volume_ratio_1d_5d": (0.80, 2.50, 0.10, "Volume ratio"),
    "volume_ratio_1d_20d": (1.00, 2.80, 0.10, "20-day volume ratio"),
    "body_to_range_ratio": (0.25, 0.70, 0.05, "Candle body ratio"),
    "ma20_slope": (0.00, 0.03, 0.0025, "MA20 slope"),
    "close_to_ma10_pct": (0.005, 0.05, 0.005, "Close to MA10"),
    "price_position_120d": (0.45, 0.90, 0.05, "120-day price position"),
    "bollinger_band_width_20d": (0.06, 0.22, 0.01, "Bollinger width"),
}

_PERIOD_TUNABLE_SPECS: dict[str, tuple[float, float, float, str]] = {
    "breakout_high_Nd": (10, 60, 5, "Breakout lookback"),
    "price_position_Nd": (60, 250, 10, "Price-position lookback"),
    "bollinger_band_width_Nd": (10, 40, 5, "Bollinger lookback"),
    "volume_ratio_1d_Nd": (5, 40, 5, "Volume baseline window"),
}


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _contains_any(text: str, terms: frozenset[str] | tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _period_tunable_key(indicator: str) -> str:
    if re.fullmatch(r"breakout_high_\d+d", indicator):
        return "breakout_high_Nd"
    if re.fullmatch(r"price_position_\d+d", indicator):
        return "price_position_Nd"
    if re.fullmatch(r"bollinger_band_width_\d+d(?:_std[0-9]+(?:p[0-9]+)?)?", indicator):
        return "bollinger_band_width_Nd"
    if re.fullmatch(r"volume_ratio_1d_\d+d", indicator):
        return "volume_ratio_1d_Nd"
    return indicator


def _requests_short_strategy(text: str) -> bool:
    return _contains_any(text, _SHORT_TERMS)


def _parse_market(market: str | MarketType) -> MarketType:
    if isinstance(market, MarketType):
        return market
    try:
        return MarketType(market)
    except ValueError as exc:
        raise ValueError(f"Unsupported market: {market}") from exc


def _infer_category(text: str) -> StrategyCategory:
    if _contains_any(text, _ROTATION_TERMS):
        return StrategyCategory.ROTATION
    if _contains_any(text, _EVENT_TERMS):
        return StrategyCategory.EVENT
    if _contains_any(text, _REVERSAL_TERMS):
        return StrategyCategory.REVERSAL
    return StrategyCategory.TREND


def _infer_style(text: str, category: StrategyCategory) -> str:
    if category == StrategyCategory.TREND and _contains_any(text, _BREAKOUT_TERMS):
        return "breakout"
    if category == StrategyCategory.TREND and _contains_any(text, _PULLBACK_TERMS):
        return "pullback"
    return category.value


def _make_strategy_id(text: str, category: StrategyCategory, style: str) -> str:
    ascii_slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if ascii_slug:
        return f"{ascii_slug[:40].strip('_')}_v1"
    return f"draft_{category.value}_{style}_v1"


def _make_strategy_name(category: StrategyCategory, style: str) -> str:
    labels = {
        "breakout": "Breakout Draft",
        "pullback": "Trend Pullback Draft",
        StrategyCategory.REVERSAL.value: "Mean Reversion Draft",
        StrategyCategory.ROTATION.value: "Sector Rotation Draft",
        StrategyCategory.EVENT.value: "Event Driven Draft",
    }
    return labels.get(style, f"{category.value.title()} Draft")


def _tags_for(category: StrategyCategory, style: str, text: str) -> list[str]:
    tags = [category.value]
    if style != category.value:
        tags.append(style)
    if _contains_any(text, _VOLUME_TERMS):
        tags.append("volume")
    if _contains_any(text, _DRAWDOWN_TERMS):
        tags.append("risk_control")
    return tags


def _preferred_regimes(category: StrategyCategory, style: str) -> list[str]:
    if category == StrategyCategory.REVERSAL:
        return ["range_bound", "oversold_rebound"]
    if category == StrategyCategory.ROTATION:
        return ["sector_rotation", "trending_up"]
    if category == StrategyCategory.EVENT:
        return ["event_driven"]
    if style == "breakout":
        return ["trending_up", "euphoria"]
    return ["trending_up"]


def _universe_for(market: MarketType) -> UniverseConfig:
    if market == MarketType.US:
        return UniverseConfig(market=["us"])
    if market == MarketType.HK:
        return UniverseConfig(market=["hk"])
    return UniverseConfig(market=["a_share_main"])


def _market_rules_for(market: MarketType) -> dict[str, MarketRuleConfig]:
    if market == MarketType.A_SHARE:
        return {
            "a_share": MarketRuleConfig(
                t_plus_1=True,
                limit_up_down=True,
                suspension=True,
            )
        }
    return {}


def _extract_percent_after(text: str, anchors: tuple[str, ...]) -> float | None:
    for anchor in anchors:
        idx = text.find(anchor)
        if idx < 0:
            continue
        fragment = text[idx : idx + 32]
        match = re.search(r"(\d+(?:\.\d+)?)\s*%", fragment)
        if match:
            return float(match.group(1)) / 100
        match = re.search(r"0\.\d+", fragment)
        if match:
            return float(match.group(0))
    return None


def _extract_holding_days(text: str) -> int | None:
    match = re.search(r"(?:持有|holding|hold)\s*(\d{1,2})\s*(?:天|日|days?)", text)
    if not match:
        match = re.search(r"(\d{1,2})\s*(?:天|日|days?)", text)
    if not match:
        return None
    return max(1, min(60, int(match.group(1))))


def _exit_triggers_for(text: str) -> list[StrategyCondition]:
    if not _contains_any(text, _EXIT_SIGNAL_TERMS):
        return []
    if ma_period := _extract_exit_ma_period(text):
        return [StrategyCondition(indicator=f"close_below_ma{ma_period}", op="==", value=True)]
    if "rsi" in text or "过热" in text:
        return [StrategyCondition(indicator="rsi_14", op=">", value=75)]
    return [StrategyCondition(indicator="close_below_ma10", op="==", value=True)]


def _extract_exit_ma_period(text: str) -> int | None:
    patterns = (
        r"(?:跌破|破位|below|breaks?\s+below)\s*(?:ma|均线)?\s*(\d{1,3})\s*(?:日线|日均线|均线|day|d)?",
        r"(?:ma|均线)\s*(\d{1,3}).{0,12}(?:卖出|退出|exit|sell)",
    )
    for pattern in patterns:
        if match := re.search(pattern, text):
            period = int(match.group(1))
            if 1 <= period <= 250:
                return period
    return None


def _next_version_id(strategy_id: str) -> str:
    match = re.fullmatch(r"(.+)_v(\d+)", strategy_id)
    if not match:
        return f"{strategy_id}_v2"
    base, version = match.groups()
    return f"{base}_v{int(version) + 1}"


def _append_revision_note(description: str, instruction: str) -> str:
    description = description.strip()
    note = f"Revision instruction: {instruction.strip()}"
    if not description:
        return note
    return f"{description}\n\n{note}"


def _ensure_condition(
    conditions: list[StrategyCondition],
    indicator: str,
    op: ConditionOp,
    value: float | bool,
) -> None:
    existing = next((c for c in conditions if c.indicator == indicator), None)
    if existing is not None:
        existing.op = op
        existing.value = value
        return
    conditions.append(StrategyCondition(indicator=indicator, op=op, value=value))


def _ensure_filter(
    filters: list[StrategyCondition],
    indicator: str,
    op: ConditionOp,
    value: float | bool,
) -> None:
    _ensure_condition(filters, indicator, op, value)


def _tighten_conditions(conditions: list[StrategyCondition]) -> None:
    for condition in conditions:
        if not isinstance(condition.value, (int, float)) or isinstance(condition.value, bool):
            continue
        if condition.op in {">", ">="}:
            condition.value = round(float(condition.value) * 1.12, 6)
        elif condition.op in {"<", "<="}:
            condition.value = round(float(condition.value) * 0.90, 6)


def _loosen_conditions(conditions: list[StrategyCondition]) -> None:
    for condition in conditions:
        if not isinstance(condition.value, (int, float)) or isinstance(condition.value, bool):
            continue
        if condition.op in {">", ">="}:
            condition.value = round(float(condition.value) * 0.90, 6)
        elif condition.op in {"<", "<="}:
            condition.value = round(float(condition.value) * 1.12, 6)


def _scale_stop_loss(
    stop_loss: StopLossConfig,
    *,
    factor: float,
    floor: float | None = None,
    cap: float | None = None,
) -> None:
    if stop_loss.value is None:
        return
    value = float(stop_loss.value) * factor
    if floor is not None:
        value = max(floor, value)
    if cap is not None:
        value = min(cap, value)
    stop_loss.value = round(value, 6)


def _scale_take_profit(
    take_profit: TakeProfitConfig,
    *,
    factor: float,
    cap: float,
) -> None:
    if take_profit.value is None:
        return
    take_profit.value = round(min(cap, float(take_profit.value) * factor), 6)


def _merge_tunables(
    current: list[TunableParam],
    generated: list[TunableParam],
) -> list[TunableParam]:
    by_target = {param.target: param for param in current}
    for param in generated:
        by_target.setdefault(param.target, param)
    return list(by_target.values())


def _merge_conditions(
    current: list[StrategyCondition],
    generated: list[StrategyCondition],
) -> list[StrategyCondition]:
    by_indicator = {condition.indicator: condition for condition in current}
    for condition in generated:
        by_indicator[condition.indicator] = condition
    return list(by_indicator.values())
