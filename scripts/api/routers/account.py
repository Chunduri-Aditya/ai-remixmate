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
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from scripts.api import jobs as job_store
from scripts.api.auth import get_user_id
from scripts.core import crates, feedback_store
from scripts.core import paths as paths_mod

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/account", tags=["account"])


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
    """Zip of mix manifests + crates + favorites for the acting user."""
    uid = get_user_id()
    library = crates.export_user_library(uid)
    manifests = _collect_manifests(paths_mod.OUTPUTS_DIR / uid)

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
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="remixmate-export.zip"'},
    )


@router.delete("")
@router.delete("/")
def delete_account():
    """Purge the acting user's rows, files, and (best-effort) Supabase user."""
    uid = get_user_id()
    if not uid or uid == "*":
        raise HTTPException(400, detail="invalid user")

    user_out = paths_mod.OUTPUTS_DIR / uid
    if user_out.exists():
        shutil.rmtree(user_out, ignore_errors=True)

    jobs_removed = job_store.purge_user_jobs(uid)
    library_counts = crates.purge_user_library(uid)
    feedback_removed = feedback_store.purge_user_events(uid)
    supabase_deleted = _purge_supabase_user(uid)

    remaining_jobs = job_store.list_jobs(limit=1000, user_id=uid)
    remaining_crates = crates.export_user_library(uid)
    files_left = user_out.exists() and any(user_out.iterdir()) if user_out.exists() else False

    return {
        "user_id": uid,
        "jobs_removed": jobs_removed,
        "library": library_counts,
        "feedback_removed": feedback_removed,
        "supabase_deleted": supabase_deleted,
        "clean": len(remaining_jobs) == 0
        and not remaining_crates["crates"]
        and not remaining_crates["favorites"]
        and not files_left,
    }
