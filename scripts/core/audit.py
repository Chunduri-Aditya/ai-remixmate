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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

_AUDIT_PATH = Path("data/audit.jsonl")


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
    user      : Who initiated it (for future multi-user support).
    job_id    : Associated job ID (for correlation with job queue).
    metadata  : Arbitrary extra context (sizes, durations, params).
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "epoch": time.time(),
        "action": action,
        "resource": resource,
        "user": user or "local",
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
