"""
scripts/api/routers/account.py — Account export + deletion (Phase 8).

Behind ``require_user`` (mounted on the gated aggregator).
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from scripts.api import jobs as job_store
from scripts.api.auth import get_user_id
from scripts.core import audit, crates, feedback_store
from scripts.core import paths as paths_mod

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/account", tags=["account"])

# jobs.json carries job metadata only: no result (output paths) and no
# error/message text (exception strings can name server paths).
_JOB_EXPORT_FIELDS = (
    "job_id", "job_type", "status", "progress",
    "created_at", "started_at", "finished_at", "meta",
)


def _supabase_configured() -> bool:
    """True when both Supabase admin env vars are set (same test as the purge)."""
    return bool(
        os.environ.get("SUPABASE_URL", "").strip()
        and os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )


def _purge_supabase_user(user_id: str) -> bool:
    """Best-effort Supabase Admin delete when SUPABASE_SERVICE_ROLE_KEY is set."""
    base = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not base or not key:
        logger.info("Supabase user delete stubbed (no service role key) uid=%s", user_id)
        return False
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{base}/auth/v1/admin/users/{user_id}",
            method="DELETE",
            headers={
                "Authorization": f"Bearer {key}",
                "apikey": key,
            },
        )
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return 200 <= getattr(resp, "status", 200) < 300
    except Exception as exc:  # noqa: BLE001
        logger.warning("Supabase user delete failed uid=%s: %s", user_id, exc)
        return False


def _collect_manifests(user_dir: Path) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    if not user_dir.is_dir():
        return manifests
    for path in user_dir.rglob("manifest.json"):
        try:
            manifests.append(
                {
                    "path": str(path.relative_to(user_dir)),
                    "manifest": json.loads(path.read_text(encoding="utf-8")),
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return manifests


@router.get("/export")
def export_account():
    """Zip of mix manifests, crates, favorites, feedback events and job
    metadata for the acting user."""
    uid = get_user_id()
    library = crates.export_user_library(uid)
    manifests = _collect_manifests(paths_mod.OUTPUTS_DIR / uid)
    feedback = feedback_store.export_user_events(uid)
    jobs = [
        {k: j.get(k) for k in _JOB_EXPORT_FIELDS}
        for j in job_store.list_jobs(limit=100_000, user_id=uid)
    ]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "library.json",
            json.dumps(library, indent=2, default=str),
        )
        zf.writestr(
            "manifests.json",
            json.dumps(manifests, indent=2, default=str),
        )
        zf.writestr(
            "meta.json",
            json.dumps({"user_id": uid}, indent=2),
        )
        zf.writestr(
            "feedback.json",
            json.dumps(feedback, indent=2, default=str),
        )
        zf.writestr(
            "jobs.json",
            json.dumps(jobs, indent=2, default=str),
        )
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="remixmate-export.zip"'},
    )


@router.delete("")
@router.delete("/")
def delete_account():
    """Purge the acting user's rows, files, audit lines and Supabase user.

    Jobs are cancelled and marked purged first, so a worker finishing after
    the delete neither persists its row nor writes output. Every later step
    runs even when an earlier one fails; failures are collected in ``errors``
    (step name and exception type only, never the message, which can carry
    server paths). ``clean`` is true only when every step succeeded, including
    the Supabase delete when it is configured, and nothing of the user remains.
    """
    uid = get_user_id()
    if not uid or uid == "*":
        raise HTTPException(400, detail="invalid user")

    errors: list[dict[str, str]] = []
    succeeded = 0

    def _step(name: str, fn: Callable[[], Any]) -> Any:
        nonlocal succeeded
        try:
            value = fn()
        except Exception as exc:  # noqa: BLE001 - reported in errors, purge continues
            logger.warning("Account delete step %s failed uid=%s: %s", name, uid, exc)
            errors.append({"step": name, "error": type(exc).__name__})
            return None
        succeeded += 1
        return value

    _step("cancel_jobs", lambda: job_store.cancel_user_jobs(uid))
    _step("mark_purged", lambda: job_store.mark_purged(uid))

    user_out = paths_mod.OUTPUTS_DIR / uid
    if user_out.exists():
        _step("outputs", lambda: shutil.rmtree(user_out))

    jobs_removed = _step("jobs", lambda: job_store.purge_user_jobs(uid))
    library_counts = _step("crates", lambda: crates.purge_user_library(uid))
    feedback_removed = _step("feedback", lambda: feedback_store.purge_user_events(uid))
    audit_removed = _step("audit", lambda: audit.purge_user_lines(uid))

    supabase_deleted = _purge_supabase_user(uid)
    if _supabase_configured():
        if supabase_deleted:
            succeeded += 1
        else:
            errors.append({"step": "supabase", "error": "admin user delete failed"})

    def _residual() -> bool:
        remaining_jobs = job_store.list_jobs(limit=1000, user_id=uid)
        remaining_crates = crates.export_user_library(uid)
        files_left = user_out.exists() and any(user_out.iterdir())
        return bool(
            remaining_jobs
            or remaining_crates["crates"]
            or remaining_crates["favorites"]
            or files_left
        )

    residual = _step("verify", _residual)

    clean = not errors and residual is False
    reason = None
    if errors:
        reason = "failed steps: " + ", ".join(e["step"] for e in errors)
    elif residual:
        reason = "user data remained after purge"
    if not errors:
        status = "complete" if clean else "partial"
    else:
        status = "partial" if succeeded else "failed"

    return {
        "user_id": uid,
        "jobs_removed": jobs_removed,
        "library": library_counts,
        "feedback_removed": feedback_removed,
        "audit_removed": audit_removed,
        "supabase_deleted": supabase_deleted,
        "clean": clean,
        "status": status,
        "reason": reason,
        "errors": errors,
    }
