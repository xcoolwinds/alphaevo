"""Structured research event logger for the evolution pipeline.

Captures the Agent's reasoning process: hypotheses, observations,
diagnoses, insights, decisions, and results — making the evolution
transparent and auditable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "hypothesis",
    "observation",
    "diagnosis",
    "insight",
    "decision",
    "experiment",
    "result",
    "reflection",
]


class ResearchEvent(BaseModel):
    """A single research event in the Agent's reasoning chain."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    content: str
    round_num: int = 0
    strategy_id: str = ""
    data: dict = Field(default_factory=dict)


class ResearchLogger:
    """Collects structured research events during strategy evolution.

    Usage::

        log = ResearchLogger()
        log.log("hypothesis", "Volume spikes may precede price breakouts",
                round_num=1, strategy_id="trend_v1")
        log.log("result", "Win rate improved from 42% to 55%",
                round_num=1, data={"win_rate": 0.55})
        print(log.to_markdown())
    """

    _MAX_EVENTS = 500

    def __init__(self) -> None:
        self._events: list[ResearchEvent] = []

    def log(
        self,
        event_type: EventType,
        content: str,
        *,
        round_num: int = 0,
        strategy_id: str = "",
        data: dict | None = None,
    ) -> ResearchEvent:
        """Record a research event and return it."""
        event = ResearchEvent(
            event_type=event_type,
            content=content,
            round_num=round_num,
            strategy_id=strategy_id,
            data=data or {},
        )
        self._events.append(event)
        if len(self._events) > self._MAX_EVENTS:
            self._events = self._events[-self._MAX_EVENTS :]
        return event

    @property
    def events(self) -> list[ResearchEvent]:
        """All recorded events (read-only copy)."""
        return list(self._events)

    def get_events(
        self,
        *,
        round_num: int | None = None,
        event_type: EventType | None = None,
    ) -> list[ResearchEvent]:
        """Filter events by round and/or type."""
        result = self._events
        if round_num is not None:
            result = [e for e in result if e.round_num == round_num]
        if event_type is not None:
            result = [e for e in result if e.event_type == event_type]
        return result

    def get_round_summary(self, round_num: int) -> str:
        """One-line summary for a specific round."""
        events = self.get_events(round_num=round_num)
        if not events:
            return f"Round {round_num}: No events recorded"

        parts = []
        for e in events:
            if e.event_type in ("result", "decision", "insight"):
                parts.append(e.content)
        return (
            f"Round {round_num}: {' → '.join(parts)}"
            if parts
            else f"Round {round_num}: {len(events)} events"
        )

    def get_summary(self) -> str:
        """Brief summary of the entire research session.

        Useful for injecting into LLM prompts as context.
        """
        if not self._events:
            return "No research events recorded yet."

        rounds = sorted({e.round_num for e in self._events if e.round_num > 0})
        lines = []
        for r in rounds:
            lines.append(self.get_round_summary(r))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        """Export the full research log as a Markdown document."""
        if not self._events:
            return "# Research Log\n\nNo events recorded.\n"

        lines = ["# Research Log\n"]
        rounds = sorted({e.round_num for e in self._events})

        for r in rounds:
            events = self.get_events(round_num=r)
            if r == 0:
                lines.append("## Setup\n")
            else:
                lines.append(f"## Round {r}\n")

            for e in events:
                icon = _EVENT_ICONS.get(e.event_type, "📌")
                lines.append(f"- {icon} **{e.event_type}**: {e.content}")
                if e.data:
                    for k, v in e.data.items():
                        lines.append(f"  - {k}: {v}")
            lines.append("")

        return "\n".join(lines)

    def clear(self) -> None:
        """Remove all recorded events."""
        self._events.clear()

    def __len__(self) -> int:
        return len(self._events)


_EVENT_ICONS: dict[str, str] = {
    "hypothesis": "💡",
    "observation": "👁️",
    "diagnosis": "🔍",
    "insight": "🧠",
    "decision": "✅",
    "experiment": "🧪",
    "result": "📊",
    "reflection": "🪞",
}
