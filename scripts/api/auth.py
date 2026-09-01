"""
scripts/api/auth.py — Per-request / per-job user scoping.

Phase 3 of the hosted migration gives the app multi-user data isolation
without adding authentication yet. Every request and every background job
runs "as" a user id held in a contextvar; until Phase 4 wires real auth that
id is always ``DEFAULT_USER`` ("local"), so single-user behaviour stays
byte-identical.

One load-bearing rule: ``ThreadPoolExecutor`` does not copy context into its
worker threads, so ``scripts/api/jobs.py::_wrapper`` must re-``set`` this var
from the job's stamped ``meta["user_id"]`` before the task runs. Mirrors the
contextvar pattern in ``scripts/core/logging_utils.py`` (request/job trace ids).
"""

from __future__ import annotations

import contextvars

DEFAULT_USER = "local"

current_user: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_user", default=DEFAULT_USER
)


def set_user_id(user_id: str) -> None:
    """Set the acting user for the current context (request or worker thread)."""
    current_user.set(user_id)


def get_user_id() -> str:
    """Return the acting user id, or ``DEFAULT_USER`` outside any set context."""
    return current_user.get()
