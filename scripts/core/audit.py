"""
scripts/core/audit.py — Immutable JSONL audit log for all operations.

Every download, remix, deletion, and library modification is recorded
in a tamper-evident append-only log (data/audit.jsonl).

Usage:
    from scripts.core.audit import log_audit
    log_audit("download_start", resource="Anyma - Voices", metadata={"source": "youtube"})
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from scripts.core.paths import DATA_DIR

log = logging.getLogger(__name__)

# Was Path("data/audit.jsonl") — CWD-relative, so it broke whenever the process
# ran from anywhere but the repo root. Derive from DATA_DIR (honours
# REMIXMATE_DATA_DIR; pointed at a throwaway dir by the test suite).
_AUDIT_PATH = DATA_DIR / "audit.jsonl"


def _acting_user() -> str:
    """Acting user id, defaulting to "local". Imported lazily so this core
    module never pulls in the API package at import time."""
    try:
        from scripts.api.auth import get_user_id

        return get_user_id()
    except Exception:
        return "local"


def log_audit(
    action: str,
    *,
    resource: Optional[str] = None,
    user: Optional[str] = None,
    job_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append an immutable audit entry.

    Parameters
    ----------
    action    : Short verb — "download_start", "remix_complete", "library_delete", etc.
    resource  : What was acted on — song name, file path, etc.
    user      : Who initiated it. Defaults to the acting user from
                scripts.api.auth.current_user (a contextvar, "local" until Phase 4
                wires real auth), so every record is attributable per-tenant.
    job_id    : Associated job ID (for correlation with job queue).
    metadata  : Arbitrary extra context (sizes, durations, params).
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "action": action,
        "resource": resource,
        "user": user or _acting_user(),
        "job_id": job_id,
        "meta": metadata or {},
    }
    try:
        _AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:
        log.error("Audit write failed: %s", exc)


def read_audit(
    limit: int = 100,
    action_filter: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Read recent audit entries (newest first)."""
    if not _AUDIT_PATH.exists():
        return []
    entries = []
    with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if action_filter and entry.get("action") != action_filter:
                    continue
                entries.append(entry)
            except json.JSONDecodeError:
                continue
    return list(reversed(entries))[:limit]


def purge_user_lines(user_id: str) -> int:
    """Remove every audit entry recorded for ``user_id`` (account deletion).

    Atomic rewrite: surviving lines are copied verbatim into a temp file in the
    same directory, which then replaces the log in one ``os.replace``. A missing
    log is not an error. Raises on I/O failure so the caller can report it.
    Returns the number of lines removed.
    """
    if not _AUDIT_PATH.exists():
        return 0
    kept: list[str] = []
    removed = 0
    with open(_AUDIT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                kept.append(line)
                continue
            if isinstance(entry, dict) and entry.get("user") == user_id:
                removed += 1
            else:
                kept.append(line)
    if removed == 0:
        return 0
    fd, tmp = tempfile.mkstemp(dir=_AUDIT_PATH.parent, prefix=".audit.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.writelines(kept)
        os.replace(tmp, _AUDIT_PATH)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return removed
