"""
scripts/api/jobs.py — Durable async job queue for long-running tasks.

Architecture
------------
- Jobs run in a ThreadPoolExecutor (CPU-bound: librosa, demucs, scipy).
- Each job gets a UUID. Callers poll GET /jobs/{job_id} for status.
- Progress is updated by the task function via update_job().
- **Job state is persisted** through a SQLAlchemy Core engine (scripts.core.db):
  SQLite at data/jobs.db by default, Postgres when DATABASE_URL is set. It
  survives process restarts. On startup, any jobs marked RUNNING are rolled
  back to FAILED since they could not have completed cleanly.

Write strategy: write-through cache.
  - In-memory _jobs dict for O(1) reads inside hot request paths.
  - Every mutation flushes the changed row to the database immediately.
  - On module import the full job table is loaded back into _jobs.

Cancel & retry
--------------
- cancel_job(job_id): marks a PENDING or RUNNING job CANCELLED; the worker
  checks this flag at startup and exits cleanly.
- retry_job(job_id, fn, **kwargs): clones a FAILED or CANCELLED job into a new
  job record and submits it to the executor.

Public API (same as before — no callers need to change)
--------------------------------------------------------
    create_job(job_type, meta)          → str (job_id)
    get_job(job_id)                     → Optional[Dict]
    list_jobs(limit)                    → list[Dict]
    update_job(job_id, *, ...)          → None
    submit_job(job_id, fn, **kwargs)    → None
    cancel_job(job_id)                  → bool
    retry_job(job_id, fn, **kwargs)     → Optional[str]  (new job_id or None)
    job_to_response(job)                → JobResponse
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException
from sqlalchemy import Connection, text

from scripts.api.auth import DEFAULT_USER, current_user, get_user_id
from scripts.api.schemas import JobResponse, JobStatus, JobType
from scripts.core import db, paths
from scripts.core.logging_utils import get_logger, set_job_id
from scripts.core.paths import DATA_DIR, LIBRARY_DIR, OUTPUTS_DIR, PROJECT_ROOT

logger = get_logger(__name__)


def _relativize(text: str) -> str:
    """Strip server path prefixes from ``text`` so no absolute path is persisted.

    OUTPUTS_DIR paths become outputs relative (``u1/dj_s1/mix.wav``, the shape
    ``/outputs`` serves), LIBRARY_DIR paths become ``library/...``, anything
    else under PROJECT_ROOT becomes root relative and anything else under the
    server user's home (model caches such as ``~/.cache/torch``) becomes ``~/...``.
    DATA_DIR (which REMIXMATE_DATA_DIR may place outside the root) becomes
    ``data/...`` and the system temp dir becomes ``tmp/...``.
    """
    for base, sub, bare in (
        (OUTPUTS_DIR, "", "outputs"),
        (LIBRARY_DIR, "library/", "library"),
        (DATA_DIR, "data/", "data"),
        (PROJECT_ROOT, "", "."),
        (tempfile.gettempdir(), "tmp/", "tmp"),
        (os.path.expanduser("~"), "~/", "~"),
    ):
        prefix = str(base)
        if len(prefix) < 2:  # an unresolved ("~") or root ("/") home would eat every slash
            continue
        text = text.replace(prefix + "/", sub).replace(prefix, bare)
    return text


# A slash rooted token: "/" not preceded by a word char, ".", "~", ":", "/" or
# "-", so relative paths, ratios ("4/4") and URLs ("https://...") are left alone.
_ABS_PATH_RE = re.compile(r"(?<![\w.~:/-])/[^\s'\"`),;\]]+")


def _scrub(text: str) -> str:
    """Relativize ``text`` (see _relativize), then replace any absolute path left.

    For client facing error and message text only: tool paths such as
    ``/opt/homebrew/bin/ffmpeg`` or ``/private/tmp/...`` match no known prefix.
    Results keep the plain _relativize pass, since ``stream_url`` is a rooted
    URL path by design.
    """
    return _ABS_PATH_RE.sub("<path>", _relativize(text))


def _relativize_value(value: Any) -> Any:
    """Relativize every string inside ``value``, walking dict values and list items."""
    if isinstance(value, str):
        return _relativize(value)
    if isinstance(value, dict):
        return {k: _relativize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativize_value(v) for v in value]
    return value


def _relativize_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``result`` with every nested string relativized (see _relativize)."""
    if isinstance(result, dict):
        return _relativize_value(result)
    return result

# ---------------------------------------------------------------------------
# Database backend (SQLAlchemy Core — SQLite by default, Postgres via DATABASE_URL)
# ---------------------------------------------------------------------------

_DB_PATH = DATA_DIR / "jobs.db"


def _schema_ddl(dialect_name: str) -> str:
    """The ``jobs`` DDL for one dialect.

    Timestamps are epoch seconds from ``time.time()`` (~1.7e9, needs ~16
    significant digits). SQLite ``REAL`` is already an 8-byte double; Postgres
    ``REAL`` is float4 (~7 digits) and would collide adjacent ``created_at``
    values, scrambling ``ORDER BY created_at DESC`` — so it gets DOUBLE
    PRECISION.
    """
    ts = "DOUBLE PRECISION" if dialect_name == "postgresql" else "REAL"
    return f"""
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    job_type    TEXT NOT NULL,
    created_at  {ts} NOT NULL,
    started_at  {ts},
    finished_at {ts},
    progress    {ts} NOT NULL DEFAULT 0.0,
    message     TEXT NOT NULL DEFAULT '',
    result_json TEXT,
    error       TEXT,
    eta_sec     INTEGER,
    meta_json   TEXT NOT NULL DEFAULT '{{}}'
)
"""


# Back-compat: a couple of tests read this attribute for a local SQLite build.
_DB_SCHEMA = _schema_ddl("sqlite")


def _get_conn() -> Connection:
    """A SQLAlchemy Connection to the job store.

    Per-store SQLite file when ``DATABASE_URL`` is unset, the shared Postgres
    engine when it is set. Callers wrap this in ``closing()`` and call
    ``conn.commit()`` after writes — the same lifecycle the raw sqlite3 handle
    had. Kept as the module's mockable seam: tests patch it to inject failures.
    """
    return db.get_engine(str(_DB_PATH)).connect()


def _init_db() -> None:
    engine = db.get_engine(str(_DB_PATH))
    with engine.begin() as conn:
        conn.execute(text(_schema_ddl(engine.dialect.name)))


def _row_to_dict(row: Any) -> Dict[str, Any]:
    d = dict(row._mapping)
    d["result"] = json.loads(d.pop("result_json") or "null")
    d["meta"]   = json.loads(d.pop("meta_json") or "{}")
    return d


def _upsert_row(conn: Connection, job: Dict[str, Any]) -> None:
    conn.execute(
        text(
            """
        INSERT INTO jobs
            (job_id, status, job_type, created_at, started_at, finished_at,
             progress, message, result_json, error, eta_sec, meta_json)
        VALUES
            (:job_id, :status, :job_type, :created_at, :started_at, :finished_at,
             :progress, :message, :result_json, :error, :eta_sec, :meta_json)
        ON CONFLICT(job_id) DO UPDATE SET
            status      = excluded.status,
            started_at  = excluded.started_at,
            finished_at = excluded.finished_at,
            progress    = excluded.progress,
            message     = excluded.message,
            result_json = excluded.result_json,
            error       = excluded.error,
            eta_sec     = excluded.eta_sec,
            meta_json   = excluded.meta_json
        """
        ),
        {
            "job_id":      job["job_id"],
            "status":      _status_value(job["status"]),
            "job_type":    _type_value(job["job_type"]),
            "created_at":  job["created_at"],
            "started_at":  job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "progress":    job.get("progress", 0.0),
            "message":     job.get("message", ""),
            "result_json": json.dumps(job.get("result")),
            "error":       job.get("error"),
            "eta_sec":     job.get("eta_sec"),
            "meta_json":   json.dumps(job.get("meta", {})),
        },
    )
    conn.commit()


# ---------------------------------------------------------------------------
# In-memory state + executor
# ---------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_cancelled: set[str] = set()          # job IDs that were cancelled before starting
_purged: set[str] = set()             # job IDs whose owner was purged (see mark_purged)
_queued: set[str] = set()             # job IDs handed to the executor, not yet picked up by a worker
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="remixmate-worker")
_MAX_ACTIVE_JOBS = 4  # Hard cap on one user's concurrent background jobs

# SSE hook — registered by scripts.api.routers.events on startup.
# Signature: (event_type: str, job_dict: dict) -> None  (sync callable)
_sse_hook: Optional[Callable] = None


def register_sse_hook(fn: Callable) -> None:
    """Register a sync callable that receives (event_type, job_dict) after each mutation."""
    global _sse_hook
    _sse_hook = fn


def _emit(event_type: str, job: Dict[str, Any]) -> None:
    """Fire the SSE hook if registered. Never raises — SSE is best-effort."""
    if _sse_hook is not None:
        try:
            _sse_hook(event_type, _job_to_response_dict(job))
        except Exception:  # noqa: BLE001
            pass


def _status_value(value: Any) -> str:
    """Normalize a status (enum, 'JobStatus.X', or plain string) to its enum value, e.g. 'done'."""
    if isinstance(value, JobStatus):
        return value.value
    return str(value or "pending").split(".")[-1].lower()


def _type_value(value: Any) -> str:
    """Normalize a job type (enum, 'JobType.X', or plain string) to its enum value, e.g. 'download'."""
    if isinstance(value, JobType):
        return value.value
    return str(value or "").split(".")[-1].lower()


def _norm_status(value: Any) -> str:
    """Normalize a status to the frontend contract (uppercase, DONE → COMPLETED)."""
    s = _status_value(value).upper()
    return "COMPLETED" if s == "DONE" else s


def _norm_type(value: Any) -> str:
    """Normalize a job type (enum, 'JobType.X', or plain string) to lowercase."""
    return _type_value(value)


def _job_to_response_dict(job: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize a job dict to a frontend-friendly payload."""
    import datetime as _dt
    def _iso(ts):
        if ts is None:
            return None
        return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat(timespec="milliseconds")

    return {
        "job_id":     job["job_id"],
        "status":     _norm_status(job.get("status")),
        "type":       _norm_type(job.get("job_type")),
        "progress":   round((job.get("progress", 0.0) or 0.0) * 100),  # 0–100 for frontend
        "message":    job.get("message", ""),
        "created_at": _iso(job.get("created_at")),
        "updated_at": _iso(job.get("finished_at") or job.get("started_at") or job.get("created_at")),
        "result":     job.get("result"),
        "error":      job.get("error"),
        "meta":       job.get("meta", {}),
    }


def _load_from_db() -> None:
    """
    Called once at module import. Restores job history from SQLite.
    Any job recorded as RUNNING is marked FAILED — it could not have completed
    cleanly if the process restarted.
    """
    try:
        _init_db()
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                text("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 500")
            ).fetchall()

        for row in rows:
            job = _row_to_dict(row)

            # Normalize legacy rows that stored 'JobStatus.X' / 'JobType.X' strings
            job["status"]   = _status_value(job["status"])
            job["job_type"] = _type_value(job["job_type"])

            # Roll back orphaned RUNNING/PENDING jobs — the executor queue
            # died with the previous process, so they can never complete.
            if job["status"] in (JobStatus.RUNNING.value, JobStatus.PENDING.value):
                job["status"]  = JobStatus.FAILED.value
                job["error"]   = "Process restarted before job could finish"
                job["message"] = "Failed: process restarted"
                with closing(_get_conn()) as conn:
                    _upsert_row(conn, job)

            _jobs[job["job_id"]] = job

        logger.info("Job store loaded from SQLite", extra={"count": len(_jobs)})
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: fallback to empty in-memory store if DB is unavailable
        logger.warning(f"Could not load jobs from SQLite, starting fresh: {exc}")


_load_from_db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_job(job_type: JobType, meta: Optional[Dict] = None) -> str:
    """Register a new job, persist it, and return its ID."""
    job_id = str(uuid.uuid4())
    # Stamp the acting user so the worker thread can re-establish context
    # (ThreadPoolExecutor does not copy contextvars) and list_jobs can scope.
    meta = dict(meta or {})
    meta.setdefault("user_id", get_user_id())
    job: Dict[str, Any] = {
        "job_id":      job_id,
        "status":      JobStatus.PENDING,
        "job_type":    job_type,
        "created_at":  time.time(),
        "started_at":  None,
        "finished_at": None,
        "progress":    0.0,
        "message":     "Queued",
        "result":      None,
        "error":       None,
        "eta_sec":     None,
        "meta":        meta,
    }
    with _lock:
        _jobs[job_id] = job
        try:
            with closing(_get_conn()) as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not persist job to SQLite: {exc}")

    logger.info("Job created", extra={"job_id": job_id, "job_type": job_type})
    _emit("job_created", job)
    return job_id


def get_job(job_id: str) -> Optional[Dict]:
    """Return the job dict or None if not found."""
    return _jobs.get(job_id)


def get_job_from_db(job_id: str) -> Optional[Dict]:
    """Read one job row straight from the database, bypassing the in-memory cache.

    Covers jobs evicted from ``_jobs`` (only the newest 500 are loaded at
    startup). Returns None when no row exists.
    """
    with closing(_get_conn()) as conn:
        row = conn.execute(
            text("SELECT * FROM jobs WHERE job_id = :id"), {"id": job_id}
        ).fetchone()
    if row is None:
        return None
    job = _row_to_dict(row)
    job["status"] = _status_value(job["status"])
    job["job_type"] = _type_value(job["job_type"])
    return job


def _result_session_keys(result: Dict[str, Any]) -> set[str]:
    """Names a job result answers to: its ``session_id`` and its output dir name.

    Full renders store the bare id but write to ``dj_<id>`` / ``chain_<id>``;
    the dir name is the parent segment of ``stream_url`` or ``output``.
    """
    keys = {result.get("session_id")}
    for field in ("stream_url", "output"):
        value = result.get(field)
        if isinstance(value, str) and "/" in value:
            keys.add(value.rstrip("/").rsplit("/", 2)[-2])
    return {k for k in keys if isinstance(k, str) and k}


def mark_expired(session_id: str) -> int:
    """Set ``result.expired = True`` on every job whose session matches ``session_id``.

    Called by the retention sweep after a session's output directory is
    removed, with the removed directory name. A job matches on its
    ``result.session_id`` or its output directory name (see
    _result_session_keys). Updates the database row and the in-memory copy.
    Returns the number of jobs flagged.
    """
    flagged = 0
    with _lock:
        with closing(_get_conn()) as conn:
            rows = conn.execute(
                text("SELECT job_id, result_json FROM jobs WHERE result_json IS NOT NULL")
            ).fetchall()
            for row in rows:
                result = json.loads(row.result_json or "null")
                if not isinstance(result, dict) or session_id not in _result_session_keys(result):
                    continue
                result["expired"] = True
                conn.execute(
                    text("UPDATE jobs SET result_json = :r WHERE job_id = :id"),
                    {"r": json.dumps(result), "id": row.job_id},
                )
                job = _jobs.get(row.job_id)
                if job is not None:
                    job["result"] = {**(job.get("result") or {}), "expired": True}
                flagged += 1
            conn.commit()
    return flagged


def delete_job(job_id: str) -> bool:
    """Remove a job from memory and the database. Returns True if it existed."""
    with _lock:
        job = _jobs.pop(job_id, None)
        try:
            with closing(_get_conn()) as conn:
                conn.execute(text("DELETE FROM jobs WHERE job_id = :id"), {"id": job_id})
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not delete job from SQLite: {exc}")
        return job is not None


def purge_user_jobs(user_id: str) -> int:
    """Delete every job owned by ``user_id``. Returns the number removed."""
    with _lock:
        ids = [jid for jid, j in _jobs.items() if _job_user(j) == user_id]
        for jid in ids:
            _jobs.pop(jid, None)
        deleted = 0
        try:
            with closing(_get_conn()) as conn:
                rows = conn.execute(text("SELECT job_id, meta_json FROM jobs")).fetchall()
                for row in rows:
                    meta = json.loads(row.meta_json or "{}")
                    if (meta.get("user_id") or DEFAULT_USER) == user_id:
                        conn.execute(
                            text("DELETE FROM jobs WHERE job_id = :id"),
                            {"id": row.job_id},
                        )
                        deleted += 1
                conn.commit()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not purge user jobs from SQLite: {exc}")
            deleted = max(deleted, len(ids))
        return deleted if deleted else len(ids)


def mark_job_takedown(job_id: str) -> Optional[Dict]:
    """Admin takedown: mark job failed/takedown and return the job snapshot."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        job["status"] = JobStatus.FAILED
        job["message"] = "Removed by admin takedown"
        job["error"] = "admin_takedown"
        job["finished_at"] = time.time()
        job["result"] = None
        try:
            with closing(_get_conn()) as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not persist takedown to SQLite: {exc}")
        return dict(job)


def _enum_forms(value: str) -> list[str]:
    """
    The two lowercased spellings a single enum value can have on disk.

    Rows written at different times persisted either the value ('dj_remix') or
    the repr ('JobType.DJ_REMIX'), so a lookup has to accept both.
    """
    bare = value.split(".")[-1].lower()
    return [bare, f"jobtype.{bare}", f"jobstatus.{bare}"]


def _query_db(job_type: Optional[str], status: Optional[str]) -> list[Dict]:
    """
    Read matching jobs straight from SQLite.

    The in-memory dict only holds the newest 500 rows (see _load_from_db), so a
    filtered lookup answered from memory silently omits anything older — which is
    how Mix Vault came to hide completed mixes. SQLite is the full record.
    """
    clauses: list[str] = []
    params: dict[str, str] = {}

    for column, value in (("job_type", job_type), ("status", status)):
        if value is None:
            continue
        forms = _enum_forms(value)
        keys = [f"{column}_{i}" for i in range(len(forms))]
        clauses.append(f"lower({column}) IN ({', '.join(':' + k for k in keys)})")
        params.update(zip(keys, forms))

    sql = "SELECT * FROM jobs"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC"

    try:
        with closing(_get_conn()) as conn:
            rows = conn.execute(text(sql), params).fetchall()
    except Exception as exc:  # noqa: BLE001
        # Fall back to the in-memory view rather than failing the request; the
        # caller still gets the newest jobs, just not the full history.
        logger.warning(f"Filtered job query fell back to memory: {exc}")
        with _lock:
            return list(_jobs.values())

    jobs = []
    for row in rows:
        job = _row_to_dict(row)
        job["status"] = _status_value(job["status"])
        job["job_type"] = _type_value(job["job_type"])
        jobs.append(job)
    return jobs


def _job_user(job: Dict) -> str:
    """Owner of a job row. Legacy rows (no stamped meta.user_id) are 'local'."""
    return (job.get("meta") or {}).get("user_id") or DEFAULT_USER


def list_jobs(
    limit: int = 50,
    job_type: Optional[str] = None,
    status: Optional[str] = None,
    user_id: Optional[str] = None,
) -> list[Dict]:
    """
    Return the most recent jobs, newest first.

    ``job_type``/``status`` filter before the limit is applied, so callers after a
    specific slice (Mix Vault wanting completed remixes) get that slice rather than
    whatever happens to sit in the newest N rows.

    Both sides of each comparison go through _type_value/_status_value because the
    store holds two serializations of the same enums — legacy rows persisted the
    repr ('JobType.DJ_REMIX', 'JobStatus.DONE') alongside newer rows holding the
    value ('dj_remix', 'done'). Comparing raw strings silently drops the legacy set.

    ``user_id`` scopes the result to one owner. ``None`` (the default) means the
    acting user from the ``current_user`` contextvar; pass ``"*"`` for the
    unscoped view (operator health/telemetry only). The scope is applied as the
    last step on both the in-memory and the SQLite path so the no-argument call
    used by the job cap is covered too.
    """
    scope = get_user_id() if user_id is None else user_id

    def _scoped(rows: list[Dict]) -> list[Dict]:
        if scope == "*":
            return rows
        return [j for j in rows if _job_user(j) == scope]

    if job_type is None and status is None:
        with _lock:
            jobs = list(_jobs.values())
        jobs = _scoped(jobs)
        return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]

    jobs = _query_db(job_type=job_type, status=status)

    # The in-memory copy is the fresher of the two for a job still being worked
    # on, so let it win where both hold the same id.
    with _lock:
        jobs = [_jobs.get(j["job_id"], j) for j in jobs]

    if job_type is not None:
        wanted_type = _type_value(job_type)
        jobs = [j for j in jobs if _type_value(j.get("job_type")) == wanted_type]

    if status is not None:
        wanted_status = _status_value(status)
        jobs = [j for j in jobs if _status_value(j.get("status")) == wanted_status]

    jobs = _scoped(jobs)

    return sorted(jobs, key=lambda j: j["created_at"], reverse=True)[:limit]


def update_job(
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    progress: Optional[float] = None,
    message: Optional[str] = None,
    result: Optional[Dict] = None,
    error: Optional[str] = None,
) -> None:
    """Thread-safe job state update — writes through to SQLite."""
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return

        if status is not None:
            job["status"] = status
        if message is not None:
            job["message"] = _scrub(message)
        if result is not None:
            job["result"] = _relativize_result(result)
        if error is not None:
            job["error"] = _scrub(error)

        if progress is not None:
            progress = round(min(1.0, max(0.0, progress)), 3)
            job["progress"] = progress

            # Compute ETA from elapsed time
            if progress > 0.0 and job.get("started_at") is not None:
                elapsed    = max(time.time() - job["started_at"], 1e-6)
                rate       = progress / elapsed
                remaining  = (1.0 - progress) / (rate + 1e-8)
                job["eta_sec"] = int(remaining)

        try:
            with closing(_get_conn()) as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not persist job update to SQLite: {exc}")

        # Emit SSE after update (still under lock to get consistent snapshot)
        current_status = _status_value(job.get("status"))
        job_snapshot = dict(job)

    # Determine SSE event type from the new status
    status_to_event = {
        "done":      "job_completed",
        "failed":    "job_failed",
        "cancelled": "job_cancelled",
    }
    sse_type = status_to_event.get(current_status, "job_updated")
    _emit(sse_type, job_snapshot)


def cancel_job(job_id: str) -> bool:
    """
    Request cancellation of a job.

    - PENDING jobs: marked CANCELLED immediately; the executor wrapper will
      skip execution if it hasn't started yet.
    - RUNNING jobs: the cancellation flag is recorded; task functions should
      check get_job(job_id)['status'] periodically for long-running loops and
      raise or return early. The status is set to CANCELLED here so polling
      clients see it immediately.
    - DONE / FAILED / CANCELLED: no-op, returns False.

    Returns True if the cancellation was recorded, False otherwise.
    """
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return False
        if _status_value(job["status"]) in ("done", "failed", "cancelled"):
            return False

        _cancelled.add(job_id)
        job["status"]      = JobStatus.CANCELLED
        job["message"]     = "Cancelled by user"
        job["finished_at"] = time.time()

        try:
            with closing(_get_conn()) as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Could not persist cancellation to SQLite: {exc}")

    logger.info("Job cancelled", extra={"job_id": job_id})
    cancelled_job = _jobs.get(job_id, {})
    _emit("job_cancelled", cancelled_job)
    return True


def is_queued(job_id: str) -> bool:
    """True when the job was submitted to the executor and no worker has started it yet.

    A PENDING record that was created but never submitted is not queued work."""
    return job_id in _queued


def _is_active(job: Dict) -> bool:
    """Counts toward the job cap: RUNNING, or PENDING and already handed to the executor."""
    status = _status_value(job["status"])
    return status == "running" or (status == "pending" and job.get("job_id") in _queued)


def _job_cap_error(active: int) -> HTTPException:
    return HTTPException(
        status_code=429,
        detail=(
            f"Too many active jobs ({active}/{_MAX_ACTIVE_JOBS}). "
            f"Wait for one to finish before submitting another."
        ),
    )


def is_cancelled(job_id: str) -> bool:
    """True when the job was cancelled. Long running tasks poll this between steps."""
    if job_id in _cancelled:
        return True
    job = _jobs.get(job_id)
    return job is not None and _status_value(job["status"]) == "cancelled"


def cancel_user_jobs(user_id: str) -> int:
    """Cancel every PENDING or RUNNING job owned by ``user_id``. Returns the count."""
    with _lock:
        ids = [
            jid for jid, j in _jobs.items()
            if _job_user(j) == user_id and _status_value(j["status"]) in ("pending", "running")
        ]
    return sum(1 for jid in ids if cancel_job(jid))


def mark_purged(user_id: str) -> None:
    """Flag every in-memory job owned by ``user_id`` as purged.

    The executor wrapper then skips running a queued job and skips persisting
    the outcome (status, result, output) of one already running, so an account
    deletion is not undone by a worker writing the row back.
    """
    with _lock:
        _purged.update(jid for jid, j in _jobs.items() if _job_user(j) == user_id)


def _discard_purged_output(user_id: str) -> None:
    """Remove ``outputs/<user_id>/`` after a purged owner's task has finished.

    DELETE /account empties that directory while the task is still running;
    a task that writes its result on the way out recreates it, so the worker
    removes it again once the task has returned.
    """
    user_out = paths.OUTPUTS_DIR / user_id
    if not user_out.exists():
        return
    try:
        shutil.rmtree(user_out)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Could not remove purged user output: {exc}")


def retry_job(job_id: str, fn: Callable, **kwargs) -> Optional[str]:
    """
    Clone a FAILED or CANCELLED job into a new job and submit it.

    Copies the job_type and meta from the original job, creates a fresh
    job record, submits it to the executor, and returns the new job_id.

    Returns None if the original job is not in a retryable state.
    """
    original = _jobs.get(job_id)
    if original is None:
        return None
    if _status_value(original["status"]) not in ("failed", "cancelled"):
        return None

    new_id = create_job(original["job_type"], {**original.get("meta", {}), "retried_from": job_id})
    submit_job(new_id, fn, **kwargs)
    logger.info("Job retried", extra={"original_job_id": job_id, "new_job_id": new_id})
    return new_id


def submit_job(job_id: str, fn: Callable, **kwargs) -> None:
    """
    Submit a job to the thread pool.
    The callable receives job_id as its first argument so it can call
    update_job() for progress reporting.

    Raises HTTPException(429), and marks the job FAILED, when its owner already
    has _MAX_ACTIVE_JOBS active jobs. The count and the reservation (_queued)
    happen under one lock: the router's _check_job_cap runs before the job
    exists, so N simultaneous submits from one user all pass it, and only this
    re-check at the moment the job becomes queued holds the cap.
    """

    def _wrapper() -> None:
        _queued.discard(job_id)

        # Respect cancellations that arrived before the worker started
        if job_id in _cancelled:
            return

        # Owner purged before the worker started: no run, no persist, no output
        if job_id in _purged:
            _purged.discard(job_id)
            return

        job = _jobs.get(job_id)
        if job is None:
            return

        # Set job ID in context for structured logging within the worker thread
        set_job_id(job_id)
        # Re-establish the acting user: this runs in an executor thread, which
        # does NOT inherit the submitting thread's contextvars. Without this
        # every per-user output path and store write falls back to "local".
        owner = job.get("meta", {}).get("user_id", DEFAULT_USER)
        current_user.set(owner)

        with _lock:
            # Re-check under the lock: a cancel or purge that landed between
            # the checks above and here must not be undone by this upsert
            if job_id in _cancelled:
                return
            if job_id in _purged:
                _purged.discard(job_id)
                return
            job["status"]     = JobStatus.RUNNING
            job["started_at"] = time.time()
            job["message"]    = "Running"
            try:
                with closing(_get_conn()) as conn:
                    _upsert_row(conn, job)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Could not persist job start to SQLite: {exc}")
            job_snapshot = dict(job)

        _emit("job_updated", job_snapshot)
        logger.info("Job started", extra={"job_id": job_id})

        try:
            result = fn(job_id=job_id, **kwargs)

            # Owner purged mid-flight: do not persist the completion, and drop
            # the output a task that never polls is_cancelled wrote on its way out
            if job_id in _purged:
                _purged.discard(job_id)
                _cancelled.discard(job_id)
                _discard_purged_output(owner)
                return

            # Check if cancelled mid-flight
            if job_id in _cancelled:
                _cancelled.discard(job_id)
                return

            with _lock:
                job["status"]      = JobStatus.DONE
                job["finished_at"] = time.time()
                job["progress"]    = 1.0
                job["message"]     = "Done"
                if result is not None:
                    job["result"] = _relativize_result(result)
                try:
                    with closing(_get_conn()) as conn:
                        _upsert_row(conn, job)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Could not persist job completion to SQLite: {exc}")
                job_snapshot = dict(job)
            _cancelled.discard(job_id)   # completed jobs no longer need the cancel flag

            # Terminal SSE frame. update_job() only fires for task-driven
            # progress calls, so without this the stream ends at the last
            # progress % and clients (job cards, toasts) never see COMPLETED.
            _emit("job_completed", job_snapshot)

            logger.info(
                "Job completed successfully",
                extra={"job_id": job_id, "duration_sec": job["finished_at"] - job["started_at"]},
            )

        except Exception as exc:  # noqa: BLE001
            # Owner purged mid-flight: do not persist the failure either
            if job_id in _purged:
                _purged.discard(job_id)
                _cancelled.discard(job_id)
                _discard_purged_output(owner)
                return

            # A cancelled task that raises on its way out stays CANCELLED
            if job_id in _cancelled:
                _cancelled.discard(job_id)
                logger.info("Cancelled job exited with an exception", extra={"job_id": job_id})
                return

            exc_traceback = traceback.format_exc()
            error = _scrub(str(exc))
            with _lock:
                job["status"]      = JobStatus.FAILED
                job["finished_at"] = time.time()
                job["error"]       = error
                job["message"]     = f"Failed: {error}"
                try:
                    with closing(_get_conn()) as conn:
                        _upsert_row(conn, job)
                except Exception as db_exc:  # noqa: BLE001
                    logger.warning(f"Could not persist job failure to SQLite: {db_exc}")
                job_snapshot = dict(job)
            _cancelled.discard(job_id)   # failed jobs no longer need the cancel flag

            _emit("job_failed", job_snapshot)

            logger.error(
                "Job failed with exception",
                extra={
                    "job_id":       job_id,
                    "error":        str(exc),
                    "traceback":    exc_traceback,
                    "duration_sec": job["finished_at"] - job["started_at"],
                },
            )

    job = _jobs.get(job_id)
    refused: Optional[int] = None
    with _lock:
        if job is not None:
            owner = _job_user(job)
            active = sum(
                1 for jid, j in _jobs.items()
                if jid != job_id and _job_user(j) == owner and _is_active(j)
            )
            if active >= _MAX_ACTIVE_JOBS:
                refused = active
                job["finished_at"] = time.time()
        if refused is None:
            _queued.add(job_id)
    if refused is not None:
        err = _job_cap_error(refused)
        update_job(job_id, status=JobStatus.FAILED, error=err.detail, message=f"Failed: {err.detail}")
        raise err
    _executor.submit(_wrapper)


def job_to_response(job: Dict) -> JobResponse:
    """Convert the raw job dict to a Pydantic response model.

    Uses _norm_status (uppercase, done→COMPLETED) to match the SSE serializer
    and the frontend TypeScript contract. Previously used _status_value which
    returned lowercase 'done' — disagreeing with SSE's 'COMPLETED'.
    """
    return JobResponse(
        job_id      = job["job_id"],
        status      = _norm_status(job["status"]),
        job_type    = _type_value(job["job_type"]),
        created_at  = job["created_at"],
        started_at  = job.get("started_at"),
        finished_at = job.get("finished_at"),
        progress    = job.get("progress", 0.0),
        message     = job.get("message", ""),
        result      = job.get("result"),
        error       = job.get("error"),
        eta_sec     = job.get("eta_sec"),
    )
