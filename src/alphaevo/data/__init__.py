"""Data layer — adapters, data management, and universe providers."""

from alphaevo.data.adapter import DataAdapter, DataManager
from alphaevo.data.universe import (
    AdapterUniverseProvider,
    CuratedUniverseProvider,
    CustomUniverseProvider,
    UniverseProvider,
)

__all__ = [
    "AdapterUniverseProvider",
    "CuratedUniverseProvider",
    "CustomUniverseProvider",
    "DataAdapter",
    "DataManager",
    "UniverseProvider",
]
