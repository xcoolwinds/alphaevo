"""Tests for research_log module-level exports."""

from alphaevo.research_log import (
    ResearchEvent,
    ResearchLogger,
    render_event,
    render_log_summary,
    render_round_header,
)


class TestResearchLogExports:
    def test_classes_importable(self):
        assert ResearchEvent is not None
        assert ResearchLogger is not None

    def test_functions_importable(self):
        assert callable(render_event)
        assert callable(render_log_summary)
        assert callable(render_round_header)

    def test_logger_from_package(self):
        """Can create a logger via the package-level import."""
        log = ResearchLogger()
        event = log.log("hypothesis", "test from package import", round_num=1)
        assert isinstance(event, ResearchEvent)
        assert len(log) == 1

    def test_all_exports(self):
        import alphaevo.research_log

        assert hasattr(alphaevo.research_log, "__all__")
        for name in alphaevo.research_log.__all__:
            assert hasattr(alphaevo.research_log, name)
