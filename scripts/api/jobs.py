"""
scripts/api/jobs.py — Durable async job queue for long-running tasks.

Architecture
------------
- Jobs run in a ThreadPoolExecutor (CPU-bound: librosa, demucs, scipy).
- Each job gets a UUID. Callers poll GET /jobs/{job_id} for status.
- Progress is updated by the task function via update_job().
- **Job state is persisted to SQLite** (data/jobs.db) so it survives process
  restarts. On startup, any jobs marked RUNNING are rolled back to FAILED since
  they could not have completed cleanly.

Write strategy: write-through cache.
  - In-memory _jobs dict for O(1) reads inside hot request paths.
  - Every mutation flushes the changed row to SQLite immediately.
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
import sqlite3
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from scripts.api.schemas import JobResponse, JobStatus, JobType
from scripts.core.logging_utils import get_logger, set_job_id
from scripts.core.paths import DATA_DIR

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# SQLite backend
# ---------------------------------------------------------------------------

_DB_PATH = DATA_DIR / "jobs.db"
_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    job_type    TEXT NOT NULL,
    created_at  REAL NOT NULL,
    started_at  REAL,
    finished_at REAL,
    progress    REAL NOT NULL DEFAULT 0.0,
    message     TEXT NOT NULL DEFAULT '',
    result_json TEXT,
    error       TEXT,
    eta_sec     INTEGER,
    meta_json   TEXT NOT NULL DEFAULT '{}'
);
"""


def _get_conn() -> sqlite3.Connection:
    """Open a thread-local SQLite connection (check_same_thread=False is safe
    because we serialize writes with _lock)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(_DB_SCHEMA)
        conn.commit()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["result"] = json.loads(d.pop("result_json") or "null")
    d["meta"]   = json.loads(d.pop("meta_json") or "{}")
    return d


def _upsert_row(conn: sqlite3.Connection, job: Dict[str, Any]) -> None:
    conn.execute(
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
        """,
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
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="remixmate-worker")

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
        with _get_conn() as conn:
            rows = conn.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 500").fetchall()

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
                with _get_conn() as conn:
                    _upsert_row(conn, job)

            _jobs[job["job_id"]] = job

        logger.info("Job store loaded from SQLite", extra={"count": len(_jobs)})
    except Exception as exc:  # noqa: BLE001
        # Non-fatal: fallback to empty in-memory store if DB is unavailable
        logger.warning("Could not load jobs from SQLite, starting fresh: %s", exc)


_load_from_db()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_job(job_type: JobType, meta: Optional[Dict] = None) -> str:
    """Register a new job, persist it, and return its ID."""
    job_id = str(uuid.uuid4())
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
        "meta":        meta or {},
    }
    with _lock:
        _jobs[job_id] = job
        try:
            with _get_conn() as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist job to SQLite: %s", exc)

    logger.info("Job created", extra={"job_id": job_id, "job_type": job_type})
    _emit("job_created", job)
    return job_id


def get_job(job_id: str) -> Optional[Dict]:
    """Return the job dict or None if not found."""
    return _jobs.get(job_id)


def list_jobs(limit: int = 50) -> list[Dict]:
    """Return the most recent jobs, newest first."""
    with _lock:
        jobs = list(_jobs.values())
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

        if status   is not None: job["status"]   = status
        if message  is not None: job["message"]  = message
        if result   is not None: job["result"]   = result
        if error    is not None: job["error"]    = error

        if progress is not None:
            progress = round(min(1.0, max(0.0, progress)), 3)
            job["progress"] = progress

            # Compute ETA from elapsed time
            if progress > 0.0 and job.get("started_at") is not None:
                elapsed    = time.time() - job["started_at"]
                rate       = progress / elapsed
                remaining  = (1.0 - progress) / (rate + 1e-8)
                job["eta_sec"] = int(remaining)

        try:
            with _get_conn() as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist job update to SQLite: %s", exc)

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
            with _get_conn() as conn:
                _upsert_row(conn, job)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not persist cancellation to SQLite: %s", exc)

    logger.info("Job cancelled", extra={"job_id": job_id})
    cancelled_job = _jobs.get(job_id, {})
    _emit("job_cancelled", cancelled_job)
    return True


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
    """

    def _wrapper() -> None:
        # Respect cancellations that arrived before the worker started
        if job_id in _cancelled:
            return

        job = _jobs.get(job_id)
        if job is None:
            return

        # Set job ID in context for structured logging within the worker thread
        set_job_id(job_id)

        with _lock:
            job["status"]     = JobStatus.RUNNING
            job["started_at"] = time.time()
            job["message"]    = "Running"
            try:
                with _get_conn() as conn:
                    _upsert_row(conn, job)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not persist job start to SQLite: %s", exc)

        logger.info("Job started", extra={"job_id": job_id})

        try:
            result = fn(job_id=job_id, **kwargs)

            # Check if cancelled mid-flight
            if job_id in _cancelled:
                return

            with _lock:
                job["status"]      = JobStatus.DONE
                job["finished_at"] = time.time()
                job["progress"]    = 1.0
                job["message"]     = "Done"
                if result is not None:
                    job["result"] = result
                try:
                    with _get_conn() as conn:
                        _upsert_row(conn, job)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not persist job completion to SQLite: %s", exc)

            logger.info(
                "Job completed successfully",
                extra={"job_id": job_id, "duration_sec": job["finished_at"] - job["started_at"]},
            )

        except Exception as exc:  # noqa: BLE001
            exc_traceback = traceback.format_exc()
            with _lock:
                job["status"]      = JobStatus.FAILED
                job["finished_at"] = time.time()
                job["error"]       = str(exc)
                job["message"]     = f"Failed: {exc}"
                try:
                    with _get_conn() as conn:
                        _upsert_row(conn, job)
                except Exception as db_exc:  # noqa: BLE001
                    logger.warning("Could not persist job failure to SQLite: %s", db_exc)

            logger.error(
                "Job failed with exception",
                extra={
                    "job_id":       job_id,
                    "error":        str(exc),
                    "traceback":    exc_traceback,
                    "duration_sec": job["finished_at"] - job["started_at"],
                },
            )

    _executor.submit(_wrapper)


def job_to_response(job: Dict) -> JobResponse:
    """Convert the raw job dict to a Pydantic response model."""
    return JobResponse(
        job_id      = job["job_id"],
        status      = _status_value(job["status"]),
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
