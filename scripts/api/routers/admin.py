"""
scripts/api/routers/admin.py — Admin takedown + user flag (Phase 8).

Routes use ``Depends(require_admin)`` locally so they stay on the gated
aggregator but reject non-admin tokens with 403.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from scripts.api import jobs as job_store
from scripts.api.auth import get_user_id, require_admin
from scripts.core import legal_store
from scripts.core import paths as paths_mod

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class FlagRequest(BaseModel):
    reason: str = Field("", max_length=2000)


def _mix_output_dir(job: dict) -> Path | None:
    """Best-effort locate the on-disk mix directory for a job."""
    result = job.get("result") or {}
    meta = job.get("meta") or {}
    owner = meta.get("user_id") or "local"
    for key in ("output_dir", "session_dir", "out_dir"):
        raw = result.get(key) or meta.get(key)
        if raw:
            p = Path(raw)
            if p.exists():
                return p
    session = result.get("session_id") or meta.get("session_id") or job.get("job_id")
    if session:
        candidate = paths_mod.OUTPUTS_DIR / owner / str(session)
        if candidate.exists():
            return candidate
    return None


@router.delete("/mix/{job_id}")
def admin_delete_mix(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(404, detail="mix not found")

    out = _mix_output_dir(job)
    if out is not None and out.exists():
        shutil.rmtree(out, ignore_errors=True)

    updated = job_store.mark_job_takedown(job_id)
    return {
        "job_id": job_id,
        "removed_dir": str(out) if out else None,
        "status": (updated or {}).get("status"),
    }


@router.post("/user/{user_id}/flag")
def admin_flag_user(user_id: str, body: FlagRequest | None = None):
    body = body or FlagRequest()
    admin_uid = get_user_id()
    legal_store.flag_user(
        user_id,
        reason=body.reason,
        flagged_by=admin_uid,
    )
    return {"user_id": user_id, "flagged": True, "reason": body.reason}
