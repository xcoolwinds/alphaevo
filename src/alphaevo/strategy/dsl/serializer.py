"""Strategy serializer — converts Strategy objects back to YAML DSL.

Inverse of StrategyParser: Strategy → YAML string or file.
Used by the evolution pipeline to persist newly generated strategy versions.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

if TYPE_CHECKING:
    from alphaevo.models.strategy import Strategy


class StrategySerializer:
    """Serialize Strategy objects to YAML DSL format."""

    def to_dict(self, strategy: Strategy) -> dict[str, Any]:
        """Convert Strategy to a clean dict suitable for YAML serialization.

        Handles:
        - Enum → value string
        - datetime → ISO string
        - Path → string
        - Strips None values and empty collections
        - Excludes computed fields (complexity_score, family_id)
        """
        raw = strategy.model_dump(
            mode="python",
            exclude={"complexity_score"},
        )

        # Clean up meta: remove computed fields not part of DSL
        if "meta" in raw:
            raw["meta"].pop("family_id", None)
            # Only include experimental when it's True (avoid clutter)
            if not raw["meta"].get("experimental"):
                raw["meta"].pop("experimental", None)

        return cast("dict[str, Any]", self._clean(raw))

    def to_yaml(self, strategy: Strategy) -> str:
        """Serialize Strategy to a YAML string."""
        data = self.to_dict(strategy)
        return cast(
            "str",
            yaml.dump(
                data,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
                width=100,
            ),
        )

    def to_file(self, strategy: Strategy, path: Path) -> None:
        """Serialize Strategy and write to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        content = self.to_yaml(strategy)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    # ── internal helpers ─────────────────────────────────────────────

    def _clean(self, obj: Any) -> Any:
        """Recursively clean a dict/list for YAML output."""
        if isinstance(obj, dict):
            cleaned = {}
            for key, value in obj.items():
                cleaned_value = self._clean(value)
                # Skip None values and empty lists/dicts
                if cleaned_value is None:
                    continue
                if isinstance(cleaned_value, (list, dict)) and not cleaned_value:
                    continue
                cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(obj, list):
            return [self._clean(item) for item in obj]
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return obj
