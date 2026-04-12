"""Tests for FactorStore (SQLite persistence)."""

from __future__ import annotations

import pytest

from alphaevo.alpha_factory.factor_store import FactorRecord, FactorStore

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store():
    s = FactorStore(":memory:")
    yield s
    s.close()


def _make_record(name: str = "vol_ratio", ir: float = 0.5, **kwargs) -> FactorRecord:
    defaults = dict(
        name=name,
        description="Volume ratio factor",
        rationale="Detects volume expansion",
        code="def compute(df, idx): return 1.0",
        ic_mean=0.05,
        ic_std=0.1,
        ir=ir,
        monthly_win_rate=0.6,
        turnover=0.3,
    )
    defaults.update(kwargs)
    return FactorRecord(**defaults)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFactorStore:
    def test_save_and_get(self, store: FactorStore):
        rec = _make_record()
        store.save(rec)
        got = store.get("vol_ratio")
        assert got is not None
        assert got.name == "vol_ratio"
        assert got.ir == 0.5
        assert got.code == "def compute(df, idx): return 1.0"

    def test_get_nonexistent(self, store: FactorStore):
        assert store.get("no_such_factor") is None

    def test_upsert(self, store: FactorStore):
        rec = _make_record(ir=0.3)
        store.save(rec)
        # Update with higher IR
        rec2 = _make_record(ir=0.8)
        store.save(rec2)
        got = store.get("vol_ratio")
        assert got.ir == 0.8
        # Should still be one record
        assert store.count() == 1

    def test_list_all(self, store: FactorStore):
        store.save(_make_record("a", ir=0.3))
        store.save(_make_record("b", ir=0.8))
        store.save(_make_record("c", ir=0.5))
        all_factors = store.list_all()
        assert len(all_factors) == 3
        # Should be ordered by IR desc
        assert all_factors[0].name == "b"
        assert all_factors[1].name == "c"

    def test_list_by_status(self, store: FactorStore):
        store.save(_make_record("a"))
        store.save(_make_record("b", status="retired"))
        active = store.list_all(status="active")
        assert len(active) == 1
        assert active[0].name == "a"

    def test_top_factors(self, store: FactorStore):
        for i in range(5):
            store.save(_make_record(f"f{i}", ir=float(i) * 0.1))
        top = store.top_factors(limit=3)
        assert len(top) == 3
        assert top[0].ir >= top[1].ir >= top[2].ir

    def test_increment_usage(self, store: FactorStore):
        store.save(_make_record())
        assert store.get("vol_ratio").usage_count == 0
        store.increment_usage("vol_ratio")
        store.increment_usage("vol_ratio")
        assert store.get("vol_ratio").usage_count == 2

    def test_retire(self, store: FactorStore):
        store.save(_make_record())
        store.retire("vol_ratio")
        got = store.get("vol_ratio")
        assert got.status == "retired"

    def test_delete(self, store: FactorStore):
        store.save(_make_record())
        assert store.delete("vol_ratio") is True
        assert store.get("vol_ratio") is None
        assert store.delete("vol_ratio") is False  # Already deleted

    def test_count(self, store: FactorStore):
        assert store.count() == 0
        store.save(_make_record("a"))
        store.save(_make_record("b"))
        assert store.count() == 2
        store.retire("b")
        assert store.count(status="active") == 1
        assert store.count(status="retired") == 1

    def test_top_factors_excludes_retired(self, store: FactorStore):
        store.save(_make_record("good", ir=0.9))
        store.save(_make_record("retired_one", ir=1.5, status="retired"))
        top = store.top_factors(status="active")
        names = [f.name for f in top]
        assert "retired_one" not in names
        assert "good" in names
