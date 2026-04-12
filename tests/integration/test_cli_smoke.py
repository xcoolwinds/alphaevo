"""CLI smoke tests that exercise real command entrypoints."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from alphaevo import __version__
from alphaevo.cli.main import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert f"AlphaEvo v{__version__}" in result.stdout


def test_strategy_validate_builtin_yaml() -> None:
    strategy_file = Path("strategies/builtin/ma_crossover.yaml")
    result = runner.invoke(app, ["strategy", "validate", str(strategy_file)])

    assert result.exit_code == 0
    assert "Valid:" in result.stdout


def test_strategy_list_empty_db(tmp_path: Path) -> None:
    env = {
        "ALPHAEVO_DB_PATH": str(tmp_path / "alphaevo.db"),
        "ALPHAEVO_CACHE_DIR": str(tmp_path / "cache"),
    }

    result = runner.invoke(app, ["strategy", "list"], env=env)

    assert result.exit_code == 0
    assert "No strategies found" in result.stdout
