"""
tests/test_mix_retention.py — Phase 6: per-user rendered-mix retention.

Covers the failure modes the deployment plan names:

* cap exceeded  → the oldest *own* mix is evicted, newer ones kept, count == cap.
* TTL           → an artificially back-dated mix is swept, a fresh one kept.
* isolation     → sweep_dir only ever touches the root it is handed, so one
                  user's outputs/<uid>/ cannot reach another's.
* dry run       → reports what it would remove, deletes nothing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from scripts.core.mix_retention import RetentionResult, sweep_dir

_DAY = 86_400


def _mk_mix(root: Path, name: str, *, age_days: float = 0.0, size: int = 1024) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "mix.wav").write_bytes(b"\0" * size)
    (d / "CREDITS.txt").write_text("x")
    if age_days:
        old = time.time() - age_days * _DAY
        for p in (d / "mix.wav", d / "CREDITS.txt", d):
            os.utime(p, (old, old))
    return d


class TestSweepDir:
    def test_missing_root_is_noop(self, tmp_path):
        res = sweep_dir(tmp_path / "nope", max_age_days=30, max_count=50)
        assert res.removed == [] and res.kept == 0

    def test_empty_root_is_noop(self, tmp_path):
        res = sweep_dir(tmp_path, max_age_days=30, max_count=50)
        assert res.removed == [] and res.kept == 0

    def test_ttl_removes_backdated_keeps_fresh(self, tmp_path):
        _mk_mix(tmp_path, "old", age_days=40)
        _mk_mix(tmp_path, "fresh", age_days=1)
        res = sweep_dir(tmp_path, max_age_days=30, max_count=50)
        assert res.removed == ["old"]
        assert res.removed_by_ttl == ["old"]
        assert res.kept == 1
        assert not (tmp_path / "old").exists()
        assert (tmp_path / "fresh").exists()

    def test_cap_evicts_oldest_until_at_cap(self, tmp_path):
        # 5 mixes, ages 5..1 days; cap 3 → the two oldest (5d, 4d) go.
        for i, age in enumerate([5, 4, 3, 2, 1]):
            _mk_mix(tmp_path, f"m{i}", age_days=age)
        res = sweep_dir(tmp_path, max_age_days=365, max_count=3)
        assert set(res.removed) == {"m0", "m1"}
        assert res.removed_by_cap == ["m0", "m1"]
        assert res.kept == 3
        assert {p.name for p in tmp_path.iterdir()} == {"m2", "m3", "m4"}

    def test_ttl_runs_before_cap(self, tmp_path):
        _mk_mix(tmp_path, "ancient", age_days=90)     # TTL
        _mk_mix(tmp_path, "a", age_days=3)
        _mk_mix(tmp_path, "b", age_days=2)
        _mk_mix(tmp_path, "c", age_days=1)
        res = sweep_dir(tmp_path, max_age_days=30, max_count=2)
        assert res.removed_by_ttl == ["ancient"]
        assert res.removed_by_cap == ["a"]           # oldest survivor
        assert res.kept == 2
        assert {p.name for p in tmp_path.iterdir()} == {"b", "c"}

    def test_dry_run_reports_but_deletes_nothing(self, tmp_path):
        _mk_mix(tmp_path, "old", age_days=40, size=2048)
        _mk_mix(tmp_path, "keep", age_days=1)
        res = sweep_dir(tmp_path, max_age_days=30, max_count=50, dry_run=True)
        assert res.removed == ["old"]
        assert res.dry_run is True
        assert res.freed_bytes >= 2048
        assert (tmp_path / "old").exists()           # still there

    def test_freed_bytes_counts_file_sizes(self, tmp_path):
        _mk_mix(tmp_path, "old", age_days=40, size=10_000)
        res = sweep_dir(tmp_path, max_age_days=30, max_count=50)
        assert 10_000 <= res.freed_bytes <= 10_000 + 64   # + the 1-byte CREDITS

    def test_sibling_root_untouched(self, tmp_path):
        """sweep_dir only sees the root it is handed — the other user's tree
        cannot be reached even when it has evictable mixes."""
        me = tmp_path / "u1"
        them = tmp_path / "u2"
        me.mkdir()
        them.mkdir()
        _mk_mix(me, "mine_old", age_days=40)
        _mk_mix(them, "theirs_old", age_days=99)
        sweep_dir(me, max_age_days=30, max_count=1)
        assert not (me / "mine_old").exists()
        assert (them / "theirs_old").exists()

    def test_now_is_injectable(self, tmp_path):
        _mk_mix(tmp_path, "m", age_days=10)
        # pretend "now" is 20 days in the future → the 10d mix is 30d old
        future = time.time() + 20 * _DAY
        res = sweep_dir(tmp_path, max_age_days=25, max_count=50, now=future)
        assert res.removed == ["m"]


class TestSweepCurrentUser:
    def test_scoped_to_user_outputs_dir_and_config(self, tmp_path, monkeypatch):
        from scripts.api.auth import current_user
        from scripts.core import mix_retention
        from scripts.core.config import cfg

        uid = "retention_test_user"
        monkeypatch.setattr(cfg.retention, "mix_max_age_days", 30, raising=False)
        monkeypatch.setattr(cfg.retention, "mix_max_count", 2, raising=False)

        from scripts.core.paths import OUTPUTS_DIR
        user_root = OUTPUTS_DIR / uid
        for i, age in enumerate([10, 5, 1]):
            _mk_mix(user_root, f"dj_{i}", age_days=age)

        token = current_user.set(uid)
        try:
            res = mix_retention.sweep_current_user(dry_run=False)
        finally:
            current_user.reset(token)

        assert res.kept == 2
        assert res.removed == ["dj_0"]              # oldest
        assert {p.name for p in user_root.iterdir()} == {"dj_1", "dj_2"}

    def test_never_raises_on_bad_state(self, monkeypatch):
        """A retention failure must not be able to break a render."""
        from scripts.core import mix_retention

        def boom(*a, **k):
            raise RuntimeError("disk gone")

        monkeypatch.setattr(mix_retention, "sweep_dir", boom)
        res = mix_retention.sweep_current_user(dry_run=False)
        assert isinstance(res, RetentionResult)
        assert res.removed == []


@pytest.mark.integration
class TestMixRetentionEndpoints:
    @pytest.fixture(autouse=True)
    def client(self):
        pytest.importorskip("httpx")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from scripts.api.auth import current_user
        from scripts.api.routers.library import router

        app = FastAPI()
        app.include_router(router)
        token = current_user.set("endpoint_test_user")
        with TestClient(app) as c:
            self._client = c
            yield c
        current_user.reset(token)

    def test_storage_status_is_per_user(self, monkeypatch):
        from scripts.core.paths import OUTPUTS_DIR

        _mk_mix(OUTPUTS_DIR / "endpoint_test_user", "dj_x", age_days=2)
        _mk_mix(OUTPUTS_DIR / "someone_else", "dj_y", age_days=2)

        r = self._client.get("/library/mixes/storage")
        assert r.status_code == 200
        body = r.json()
        assert body["mix_count"] == 1                       # only this user's
        assert body["outputs_dir"].endswith("endpoint_test_user")

    def test_sweep_dry_run_default_deletes_nothing(self):
        from scripts.core.paths import OUTPUTS_DIR

        _mk_mix(OUTPUTS_DIR / "endpoint_test_user", "dj_old", age_days=99)
        r = self._client.post("/library/mixes/sweep")           # dry_run defaults True
        assert r.status_code == 200
        assert r.json()["dry_run"] is True
        assert r.json()["removed"] == ["dj_old"]
        assert (OUTPUTS_DIR / "endpoint_test_user" / "dj_old").exists()

    def test_sweep_commit_removes(self):
        from scripts.core.paths import OUTPUTS_DIR

        _mk_mix(OUTPUTS_DIR / "endpoint_test_user", "dj_stale", age_days=99)
        r = self._client.post("/library/mixes/sweep", params={"dry_run": False})
        assert r.status_code == 200
        assert "dj_stale" in r.json()["removed"]
        assert not (OUTPUTS_DIR / "endpoint_test_user" / "dj_stale").exists()
