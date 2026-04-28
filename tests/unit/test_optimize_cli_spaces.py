"""Tests for optimize CLI space routing."""

from __future__ import annotations

from alphaevo.cli.commands.evolution import _split_optimize_spaces


def test_default_optimize_runs_exit_and_param_spaces() -> None:
    exit_spaces, param_spaces, run_param = _split_optimize_spaces(None)

    assert exit_spaces is None
    assert param_spaces is None
    assert run_param is True


def test_entry_space_routes_only_to_param_optimizer() -> None:
    exit_spaces, param_spaces, run_param = _split_optimize_spaces(["entry"])

    assert exit_spaces == []
    assert param_spaces == ["entry"]
    assert run_param is True


def test_all_space_routes_to_both_optimizers() -> None:
    exit_spaces, param_spaces, run_param = _split_optimize_spaces(["all"])

    assert exit_spaces is None
    assert param_spaces is None
    assert run_param is True
