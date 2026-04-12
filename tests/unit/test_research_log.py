"""Tests for ResearchLogger — structured research event tracking."""

from alphaevo.research_log.logger import ResearchEvent, ResearchLogger


class TestResearchEvent:
    def test_event_creation(self):
        event = ResearchEvent(
            event_type="hypothesis",
            content="Volume spikes precede breakouts",
            round_num=1,
            strategy_id="trend_v1",
        )
        assert event.event_type == "hypothesis"
        assert event.content == "Volume spikes precede breakouts"
        assert event.round_num == 1
        assert event.strategy_id == "trend_v1"
        assert event.timestamp is not None
        assert event.data == {}

    def test_event_with_data(self):
        event = ResearchEvent(
            event_type="result",
            content="Win rate improved",
            data={"win_rate": 0.55, "signals": 74},
        )
        assert event.data["win_rate"] == 0.55
        assert event.data["signals"] == 74

    def test_event_types_validated(self):
        valid_types = [
            "hypothesis",
            "observation",
            "diagnosis",
            "insight",
            "decision",
            "experiment",
            "result",
            "reflection",
        ]
        for t in valid_types:
            event = ResearchEvent(event_type=t, content="test")
            assert event.event_type == t


class TestResearchLogger:
    def test_empty_logger(self):
        log = ResearchLogger()
        assert len(log) == 0
        assert log.events == []

    def test_log_event(self):
        log = ResearchLogger()
        event = log.log("hypothesis", "Test hypothesis", round_num=1)
        assert len(log) == 1
        assert isinstance(event, ResearchEvent)
        assert event.event_type == "hypothesis"

    def test_log_multiple_events(self):
        log = ResearchLogger()
        log.log("hypothesis", "H1", round_num=1)
        log.log("experiment", "E1", round_num=1)
        log.log("result", "R1", round_num=1, data={"score": 0.4})
        assert len(log) == 3

    def test_events_returns_copy(self):
        log = ResearchLogger()
        log.log("hypothesis", "H1")
        events = log.events
        events.clear()
        assert len(log) == 1  # Original unaffected

    def test_filter_by_round(self):
        log = ResearchLogger()
        log.log("hypothesis", "R1", round_num=1)
        log.log("result", "R1 result", round_num=1)
        log.log("hypothesis", "R2", round_num=2)
        round1 = log.get_events(round_num=1)
        assert len(round1) == 2
        round2 = log.get_events(round_num=2)
        assert len(round2) == 1

    def test_filter_by_event_type(self):
        log = ResearchLogger()
        log.log("hypothesis", "H1", round_num=1)
        log.log("result", "R1", round_num=1)
        log.log("hypothesis", "H2", round_num=2)
        hypotheses = log.get_events(event_type="hypothesis")
        assert len(hypotheses) == 2

    def test_filter_by_both(self):
        log = ResearchLogger()
        log.log("hypothesis", "H1", round_num=1)
        log.log("hypothesis", "H2", round_num=2)
        log.log("result", "R1", round_num=1)
        r1_hyp = log.get_events(round_num=1, event_type="hypothesis")
        assert len(r1_hyp) == 1
        assert r1_hyp[0].content == "H1"

    def test_get_round_summary(self):
        log = ResearchLogger()
        log.log("hypothesis", "Test hyp", round_num=1)
        log.log("result", "Score: 45%", round_num=1)
        summary = log.get_round_summary(1)
        assert "Round 1" in summary
        assert "Score: 45%" in summary

    def test_get_round_summary_empty(self):
        log = ResearchLogger()
        summary = log.get_round_summary(99)
        assert "No events" in summary

    def test_get_summary(self):
        log = ResearchLogger()
        log.log("result", "Score: 40%", round_num=1)
        log.log("decision", "Tighten RSI", round_num=1)
        log.log("result", "Score: 50%", round_num=2)
        summary = log.get_summary()
        assert "Round 1" in summary
        assert "Round 2" in summary

    def test_get_summary_empty(self):
        log = ResearchLogger()
        summary = log.get_summary()
        assert "No research events" in summary

    def test_to_markdown(self):
        log = ResearchLogger()
        log.log("hypothesis", "Volume predicts moves", round_num=1, strategy_id="t_v1")
        log.log("result", "Win rate: 55%", round_num=1, data={"win_rate": 0.55})
        md = log.to_markdown()
        assert "# Research Log" in md
        assert "## Round 1" in md
        assert "Volume predicts moves" in md
        assert "win_rate" in md

    def test_to_markdown_empty(self):
        log = ResearchLogger()
        md = log.to_markdown()
        assert "No events recorded" in md

    def test_clear(self):
        log = ResearchLogger()
        log.log("hypothesis", "H1", round_num=1)
        log.log("result", "R1", round_num=1)
        assert len(log) == 2
        log.clear()
        assert len(log) == 0

    def test_log_returns_event(self):
        log = ResearchLogger()
        event = log.log(
            "insight",
            "RSI tightening works for trend strategies",
            round_num=2,
            strategy_id="trend_v2",
            data={"confidence": 0.8},
        )
        assert event.event_type == "insight"
        assert event.strategy_id == "trend_v2"
        assert event.data["confidence"] == 0.8
