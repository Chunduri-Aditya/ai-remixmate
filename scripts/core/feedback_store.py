"""
scripts/core/feedback_store.py — Behavioral telemetry event log
(SELF_IMPROVING_DJ_RESEARCH.md §3, Stage A).

This module exists to COLLECT data, not to learn from it yet. Stage A's
entire purpose is to find out empirically whether skip/replay rate vary
enough across transitions to justify Stage B/C model work — see
SELF_IMPROVING_DJ_RESEARCH.md §0 and §5. Nothing in here scores, ranks, or
adjusts compatibility_score() / setlist_planner.py; that's explicitly out of
scope until an offline analysis (Stage B) greenlights it.

Architecture
------------
Structurally mirrors scripts/api/jobs.py's write-through SQLite + in-memory
cache pattern (data/feedback.db, DATA_DIR-relative, same connect-per-call
style). The key structural difference: feedback_events is APPEND-ONLY — an
event is never mutated in place after being logged, unlike a job's status
field, so there's no update_job()-equivalent here, no executor, and no SSE
hook (SELF_IMPROVING_DJ_RESEARCH.md §3 is explicit that playback events are
user-initiated, not server-pushed, so they don't need one).

Write strategy: write-through cache, same as jobs.py.
  - In-memory _events list for O(1) reads inside hot request paths.
  - Every log_event() call flushes the row to SQLite immediately.
  - On module import, the most recent rows are loaded back into _events
    (capped — see _MAX_IN_MEMORY; SQLite keeps the full history regardless).

Public API
----------
    log_event(event_type, song_a, song_b=None, job_id=None,
               position_sec=None, context=None) -> str (event id)
    list_events(limit=500)  -> list[Dict]   newest first
    summarize()              -> Dict         per-song / per-song-pair aggregates
                                              for GET /feedback/summary
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import defaultdict
from contextlib import closing
from typing import Any, Dict, List, Optional

from scripts.core.logging_utils import get_logger
from scripts.core.paths import DATA_DIR

logger = get_logger(__name__)

_DEFAULT_USER = "local"


def _acting_user() -> str:
    """Acting user id (contextvar, "local" until Phase 4). Lazy import so this
    core module never pulls in the API package at load time."""
    try:
        from scripts.api.auth import get_user_id

        return get_user_id()
    except Exception:
        return _DEFAULT_USER


# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_DB_PATH = DATA_DIR / "feedback.db"
_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id            TEXT PRIMARY KEY,
    event_type    TEXT NOT NULL,
    song_a        TEXT NOT NULL,
    song_b        TEXT,
    job_id        TEXT,
    position_sec  REAL,
    context_json  TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    user_id       TEXT NOT NULL DEFAULT 'local'
);
"""

# SELF_IMPROVING_DJ_RESEARCH.md §3's event_type column comment, as a
# validated set rather than a SQLite CHECK constraint (those don't survive
# schema evolution cleanly across ALTERs, and this is easier to extend).
VALID_EVENT_TYPES = frozenset({
    "play_start", "skip", "complete", "replay",
    "reorder", "rerender", "thumbs_up", "thumbs_down",
})


def _get_conn() -> sqlite3.Connection:
    """Open a thread-local SQLite connection (check_same_thread=False is safe
    because writes are serialized with _lock, same as jobs.py)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(_DB_SCHEMA)
        # Migration: a feedback.db written before Phase 3 has no user_id column.
        # ADD COLUMN with a DEFAULT backfills every existing row to "local", so
        # single-user history stays attributed to the local user.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(feedback_events)")}
        if "user_id" not in cols:
            conn.execute(
                "ALTER TABLE feedback_events ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'"
            )
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["context"] = json.loads(d.pop("context_json") or "{}")
    return d


def _insert_row(conn: sqlite3.Connection, event: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO feedback_events
            (id, event_type, song_a, song_b, job_id, position_sec, context_json,
             created_at, user_id)
        VALUES
            (:id, :event_type, :song_a, :song_b, :job_id, :position_sec, :context_json,
             :created_at, :user_id)
        """,
        {
            "id":           event["id"],
            "event_type":   event["event_type"],
            "song_a":       event["song_a"],
            "song_b":       event.get("song_b"),
            "job_id":       event.get("job_id"),
            "position_sec": event.get("position_sec"),
            "context_json": json.dumps(event.get("context") or {}),
            "created_at":   event["created_at"],
            "user_id":      event.get("user_id") or _DEFAULT_USER,
        },
    )
    conn.commit()


# ---------------------------------------------------------------------------
# In-memory state
# ---------------------------------------------------------------------------

_events: List[Dict[str, Any]] = []
_lock = threading.Lock()

# Feedback events are append-only and can accumulate fast (every play/skip/
# reorder), unlike jobs — cap the in-memory mirror so a long-running process
# doesn't grow this list unbounded. SQLite keeps the full history regardless;
# GET /feedback/summary reflects only what's in this cache, same tradeoff
# jobs.py already makes with its own LIMIT 500 load.
_MAX_IN_MEMORY = 5000


def _load_from_db() -> None:
    """Called once at module import. Restores recent event history from SQLite."""
    try:
        _init_db()
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                "SELECT * FROM feedback_events ORDER BY created_at DESC LIMIT ?",
                (_MAX_IN_MEMORY,),
            ).fetchall()
        _events.extend(_row_to_dict(row) for row in rows)
        logger.info("Feedback store loaded from SQLite", extra={"count": len(_events)})
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: fall back to an empty in-memory store if the DB is
        # unavailable — same resilience posture as jobs.py's _load_from_db().
        logger.warning(f"Could not load feedback events from SQLite, starting fresh: {exc}")


_load_from_db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def log_event(
    event_type: str,
    song_a: str,
    song_b: Optional[str] = None,
    job_id: Optional[str] = None,
    position_sec: Optional[float] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Record one behavioral telemetry event and return its id.

    Fire-and-forget from the caller's perspective for I/O failures: a broken
    SQLite write is logged and swallowed, never raised, so a feedback-pipe
    problem can never take down playback or rendering (mirrors jobs.py's
    "warn and continue" posture on every _upsert_row call).

    Raises ValueError only for a genuinely invalid event_type — that's a
    programming error in the caller, not a runtime/IO failure, so unlike
    SQLite errors it is NOT swallowed.
    """
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(
            f"Unknown event_type {event_type!r} — must be one of {sorted(VALID_EVENT_TYPES)}"
        )

    event: Dict[str, Any] = {
        "id":           str(uuid.uuid4()),
        "event_type":   event_type,
        "song_a":       song_a,
        "song_b":       song_b,
        "job_id":       job_id,
        "position_sec": position_sec,
        "context":      context or {},
        "created_at":   time.time(),
        "user_id":      _acting_user(),
    }

    with _lock:
        _events.append(event)
        if len(_events) > _MAX_IN_MEMORY:
            del _events[: len(_events) - _MAX_IN_MEMORY]
        try:
            with closing(_get_conn()) as conn:
                _insert_row(conn, event)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not persist feedback event to SQLite: {exc}")

    logger.info(
        "Feedback event logged",
        extra={"event_type": event_type, "song_a": song_a, "song_b": song_b, "job_id": job_id},
    )
    return event["id"]


def _scoped(events: List[Dict[str, Any]], user_id: Optional[str]) -> List[Dict[str, Any]]:
    """Filter to one owner. ``None`` → the acting user (contextvar); ``"*"`` →
    unscoped (operator/analysis only). Applied to the in-memory list, mirroring
    jobs.py — the SQLite mirror only feeds this list at import, so scoping the
    list is what the API actually sees. Rows with no ``user_id`` (legacy, and
    the SQLite DEFAULT backfill) count as ``"local"``."""
    scope = _acting_user() if user_id is None else user_id
    if scope == "*":
        return events
    return [e for e in events if (e.get("user_id") or _DEFAULT_USER) == scope]


def list_events(limit: int = 500, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Return the most recent events for the acting user, newest first.

    ``user_id`` overrides the scope: an explicit id, or ``"*"`` for every user.
    """
    with _lock:
        events = _scoped(list(_events), user_id)
    return sorted(events, key=lambda e: e["created_at"], reverse=True)[:limit]


def _pair_key(song_a: str, song_b: Optional[str]) -> Optional[str]:
    """
    Order-independent key for a song pair.

    A skip/replay on A→B and the same pair played B→A in a different set
    ordering are still evaluating the *same transition* from Stage B's
    perspective (SELF_IMPROVING_DJ_RESEARCH.md §5's per-pair analysis cares
    about the pairing, not direction) — so both aggregate under one key.
    Returns None for single-song events (song_b absent), which the caller
    should skip rather than bucket.
    """
    if not song_b:
        return None
    return " :: ".join(sorted((song_a, song_b)))


def summarize(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Aggregate counts per event type / per song / per song-pair, for
    GET /feedback/summary (feeds Stage B's offline analysis —
    SELF_IMPROVING_DJ_RESEARCH.md §5).

    Scoped to the acting user by default; ``user_id`` overrides (``"*"`` for
    every user, for offline analysis).

    Deliberately does no scoring or correlation here. Stage A's job is only
    to collect and expose sane aggregates; skip-rate-vs-compatibility-score
    correlation is Stage B analysis run offline against the raw event log
    (via list_events()), not something this endpoint should pre-compute or
    bake an opinion into.
    """
    with _lock:
        events = _scoped(list(_events), user_id)

    by_type: Dict[str, int] = defaultdict(int)
    by_song: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_pair: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for e in events:
        et = e["event_type"]
        by_type[et] += 1
        by_song[e["song_a"]][et] += 1
        if e.get("song_b"):
            by_song[e["song_b"]][et] += 1
            pair = _pair_key(e["song_a"], e["song_b"])
            if pair:
                by_pair[pair][et] += 1

    return {
        "total_events":    len(events),
        "by_event_type":   dict(by_type),
        "by_song":         {name: dict(counts) for name, counts in by_song.items()},
        "by_song_pair":    {pair: dict(counts) for pair, counts in by_pair.items()},
        "oldest_event_at": min((e["created_at"] for e in events), default=None),
        "newest_event_at": max((e["created_at"] for e in events), default=None),
    }
