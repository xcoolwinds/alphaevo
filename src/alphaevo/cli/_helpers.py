"""Shared CLI helpers — config and store accessors."""

from __future__ import annotations

from typing import Any


def _get_config(overrides: dict[str, Any] | None = None) -> Any:
    from alphaevo.core.config import ConfigManager

    return ConfigManager().load(cli_overrides=overrides)


def _get_store(config: Any = None) -> Any:
    from alphaevo.strategy.store import StrategyStore

    if config is None:
        config = _get_config()
    return StrategyStore(config.db_path)
