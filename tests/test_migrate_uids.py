"""
tests/test_migrate_uids.py — the one-time "local" → real-uid migration (Phase 4).

Dry run reports and changes nothing; --apply re-stamps every ``"local"`` row in
jobs / crates / feedback and moves ``outputs/local/*`` under ``outputs/<uid>/``.
Rows already owned by another uid are left alone.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from scripts.migrate_uids import _CRATE_TABLES, run

UID = "11111111-2222-3333-4444-555555555555"


@pytest.fixture
def world(tmp_path):
    """A minimal data/ + outputs/ tree with one 'local' row per store, one
    foreign row, and a rendered mix under outputs/local/."""
    data = tmp_path / "data"
    outputs = tmp_path / "outputs"
    data.mkdir()
    (outputs / "local" / "dj_abc123").mkdir(parents=True)
    (outputs / "local" / "dj_abc123" / "mix.wav").write_bytes(b"RIFF0000WAVE")
    (outputs / "other-uid" / "dj_zzz").mkdir(parents=True)  # someone else — untouched

    # jobs.db — owner lives in the meta_json blob
    jc = sqlite3.connect(str(data / "jobs.db"))
    jc.execute(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, status TEXT, job_type TEXT,"
        " created_at REAL, progress REAL, message TEXT, result_json TEXT, meta_json TEXT)"
    )
    jc.execute("INSERT INTO jobs VALUES ('j-local','done','dj_remix',1,1.0,'',NULL,?)",
               (json.dumps({"user_id": "local", "song": "x"}),))
    jc.execute("INSERT INTO jobs VALUES ('j-legacy','done','dj_remix',1,1.0,'',NULL,'{}')")
    jc.execute("INSERT INTO jobs VALUES ('j-other','done','dj_remix',1,1.0,'',NULL,?)",
               (json.dumps({"user_id": "other-uid"}),))
    jc.commit()
    jc.close()

    # crates (remixmate.db) + feedback.db — real user_id columns
    cc = sqlite3.connect(str(data / "remixmate.db"))
    for t in _CRATE_TABLES:
        cc.execute(f"CREATE TABLE {t} (x TEXT, user_id TEXT NOT NULL DEFAULT 'local')")
        cc.execute(f"INSERT INTO {t} (x, user_id) VALUES ('a','local')")
        cc.execute(f"INSERT INTO {t} (x, user_id) VALUES ('b','other-uid')")
    cc.commit()
    cc.close()

    fc = sqlite3.connect(str(data / "feedback.db"))
    fc.execute("CREATE TABLE feedback_events (id TEXT, user_id TEXT NOT NULL DEFAULT 'local')")
    fc.execute("INSERT INTO feedback_events VALUES ('e1','local')")
    fc.execute("INSERT INTO feedback_events VALUES ('e2','local')")
    fc.execute("INSERT INTO feedback_events VALUES ('e3','other-uid')")
    fc.commit()
    fc.close()

    return data, outputs


def _owner_counts(data):
    out = {}
    jc = sqlite3.connect(str(data / "jobs.db"))
    rows = jc.execute("SELECT meta_json FROM jobs").fetchall()
    jc.close()
    out["jobs_local"] = sum(1 for (m,) in rows if (json.loads(m or "{}").get("user_id") or "local") == "local")
    out["jobs_uid"] = sum(1 for (m,) in rows if json.loads(m or "{}").get("user_id") == UID)
    cc = sqlite3.connect(str(data / "remixmate.db"))
    out["crates_local"] = cc.execute("SELECT COUNT(*) FROM crates WHERE user_id='local'").fetchone()[0]
    out["crates_uid"] = cc.execute("SELECT COUNT(*) FROM crates WHERE user_id=?", (UID,)).fetchone()[0]
    cc.close()
    fc = sqlite3.connect(str(data / "feedback.db"))
    out["fb_local"] = fc.execute("SELECT COUNT(*) FROM feedback_events WHERE user_id='local'").fetchone()[0]
    out["fb_uid"] = fc.execute("SELECT COUNT(*) FROM feedback_events WHERE user_id=?", (UID,)).fetchone()[0]
    fc.close()
    return out


def test_dry_run_changes_nothing(world, capsys):
    data, outputs = world
    run(UID, apply=False, data_dir=data, outputs_dir=outputs)

    c = _owner_counts(data)
    assert c["jobs_local"] == 2 and c["jobs_uid"] == 0   # j-local + j-legacy still local
    assert c["crates_local"] == 1 and c["crates_uid"] == 0
    assert c["fb_local"] == 2 and c["fb_uid"] == 0
    assert (outputs / "local" / "dj_abc123" / "mix.wav").exists()
    assert "Dry run" in capsys.readouterr().out


def test_apply_restamps_local_and_moves_outputs(world):
    data, outputs = world
    run(UID, apply=True, data_dir=data, outputs_dir=outputs)

    c = _owner_counts(data)
    # every 'local' row (incl. the legacy blank-meta job) now belongs to UID
    assert c["jobs_local"] == 0 and c["jobs_uid"] == 2
    assert c["crates_local"] == 0 and c["crates_uid"] == 1
    assert c["fb_local"] == 0 and c["fb_uid"] == 2

    # foreign rows untouched
    jc = sqlite3.connect(str(data / "jobs.db"))
    assert json.loads(jc.execute("SELECT meta_json FROM jobs WHERE job_id='j-other'").fetchone()[0])["user_id"] == "other-uid"
    jc.close()

    # the mix moved; outputs/local/ is gone; the other user's dir is intact
    assert (outputs / UID / "dj_abc123" / "mix.wav").exists()
    assert not (outputs / "local").exists()
    assert (outputs / "other-uid" / "dj_zzz").is_dir()


def test_apply_refuses_on_a_colliding_output_dir(world):
    data, outputs = world
    (outputs / UID / "dj_abc123").mkdir(parents=True)  # already there → collision
    with pytest.raises(SystemExit):
        run(UID, apply=True, data_dir=data, outputs_dir=outputs)
    # nothing moved, nothing re-stamped (the raise happens before any write)
    assert (outputs / "local" / "dj_abc123" / "mix.wav").exists()
    assert _owner_counts(data)["crates_local"] == 1


def test_rejects_local_or_empty_uid(world):
    data, outputs = world
    for bad in ("", "local"):
        with pytest.raises(SystemExit):
            run(bad, apply=False, data_dir=data, outputs_dir=outputs)
