"""
tests/test_feedback_store.py — scripts/core/feedback_store.py
(SELF_IMPROVING_DJ_RESEARCH.md §3, Stage A).

Each test gets its own SQLite file (via the `fresh_store` fixture, which
patches _DB_PATH and clears the module's in-memory _events list) so events
logged in one test can't leak into another through the shared module-level
cache — the same isolation problem jobs.py's tests solve by mocking
_get_conn/_upsert_row entirely; here we instead point at a real tmp_path
DB, since feedback_store's simpler append-only writes make that cheap and
it lets summarize()/persistence be tested against real SQLite.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    from scripts.core import feedback_store as fs

    monkeypatch.setattr(fs, "_DB_PATH", tmp_path / "feedback.db")
    fs._events.clear()
    fs._init_db()
    yield fs
    fs._events.clear()


class TestLogEvent:
    def test_creates_entry_with_uuid(self, fresh_store):
        event_id = fresh_store.log_event("play_start", "Song A", "Song B")
        assert isinstance(event_id, str) and len(event_id) == 36  # uuid4 string form

    def test_invalid_event_type_raises(self, fresh_store):
        with pytest.raises(ValueError, match="Unknown event_type"):
            fresh_store.log_event("not_a_real_type", "Song A")

    def test_single_song_event_leaves_song_b_none(self, fresh_store):
        fresh_store.log_event("thumbs_up", "Song A")
        events = fresh_store.list_events()
        assert events[0]["song_a"] == "Song A"
        assert events[0]["song_b"] is None

    def test_context_round_trips(self, fresh_store):
        fresh_store.log_event(
            "skip", "Song A", "Song B",
            job_id="job-123", position_sec=12.5,
            context={"transition_bars": 16, "preset": "club"},
        )
        e = fresh_store.list_events()[0]
        assert e["job_id"] == "job-123"
        assert e["position_sec"] == 12.5
        assert e["context"] == {"transition_bars": 16, "preset": "club"}

    def test_missing_context_defaults_to_empty_dict(self, fresh_store):
        fresh_store.log_event("play_start", "Song A")
        assert fresh_store.list_events()[0]["context"] == {}

    def test_io_failure_is_swallowed_not_raised(self, fresh_store, monkeypatch):
        """A broken SQLite write must never take down the caller — playback/
        rendering shouldn't fail because telemetry failed to persist."""
        def _boom(*a, **kw):
            raise RuntimeError("disk exploded")

        monkeypatch.setattr(fresh_store, "_get_conn", _boom)
        event_id = fresh_store.log_event("play_start", "Song A")  # must not raise
        assert event_id  # still recorded in memory even though the DB write failed
        assert fresh_store.list_events()[0]["song_a"] == "Song A"


class TestListEvents:
    def test_newest_first(self, fresh_store):
        fresh_store.log_event("play_start", "First")
        fresh_store.log_event("play_start", "Second")
        fresh_store.log_event("play_start", "Third")
        names = [e["song_a"] for e in fresh_store.list_events()]
        assert names == ["Third", "Second", "First"]

    def test_respects_limit(self, fresh_store):
        for i in range(5):
            fresh_store.log_event("play_start", f"Song {i}")
        assert len(fresh_store.list_events(limit=2)) == 2

    def test_empty_store_returns_empty_list(self, fresh_store):
        assert fresh_store.list_events() == []


class TestPairKey:
    def test_order_independent(self, fresh_store):
        assert fresh_store._pair_key("A", "B") == fresh_store._pair_key("B", "A")

    def test_none_song_b_returns_none(self, fresh_store):
        assert fresh_store._pair_key("A", None) is None


class TestSummarize:
    def test_empty_store(self, fresh_store):
        s = fresh_store.summarize()
        assert s["total_events"] == 0
        assert s["by_event_type"] == {}
        assert s["oldest_event_at"] is None
        assert s["newest_event_at"] is None

    def test_counts_by_event_type(self, fresh_store):
        fresh_store.log_event("play_start", "A", "B")
        fresh_store.log_event("skip", "A", "B")
        fresh_store.log_event("skip", "C", "D")
        s = fresh_store.summarize()
        assert s["total_events"] == 3
        assert s["by_event_type"] == {"play_start": 1, "skip": 2}

    def test_by_song_counts_both_sides_of_a_pair(self, fresh_store):
        fresh_store.log_event("complete", "A", "B")
        s = fresh_store.summarize()
        assert s["by_song"]["A"]["complete"] == 1
        assert s["by_song"]["B"]["complete"] == 1

    def test_by_song_pair_aggregates_regardless_of_direction(self, fresh_store):
        fresh_store.log_event("skip", "A", "B")
        fresh_store.log_event("skip", "B", "A")   # same pair, reversed order
        s = fresh_store.summarize()
        assert len(s["by_song_pair"]) == 1
        pair_counts = next(iter(s["by_song_pair"].values()))
        assert pair_counts["skip"] == 2

    def test_single_song_events_excluded_from_by_song_pair(self, fresh_store):
        fresh_store.log_event("thumbs_up", "A")
        s = fresh_store.summarize()
        assert s["by_song_pair"] == {}
        assert s["by_song"]["A"]["thumbs_up"] == 1


class TestPersistence:
    def test_events_survive_reload_from_sqlite(self, fresh_store):
        """Simulates a process restart: clear the in-memory cache and reload
        from the same DB file, same as _load_from_db() does at import time."""
        fresh_store.log_event("play_start", "A", "B", context={"k": "v"})
        fresh_store.log_event("complete", "A", "B")

        fresh_store._events.clear()
        fresh_store._load_from_db()

        events = fresh_store.list_events()
        assert len(events) == 2
        assert {e["event_type"] for e in events} == {"play_start", "complete"}
        loaded_with_context = next(e for e in events if e["event_type"] == "play_start")
        assert loaded_with_context["context"] == {"k": "v"}
