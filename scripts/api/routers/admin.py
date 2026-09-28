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
from scripts.api.schemas import JobStatus
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


def _mix_output_dir(job: dict, root: Path) -> Path | None:
    """Locate a mix's session directory, only ever under ``root`` (OUTPUTS_DIR).

    The real layout is ``<root>/<owner>/<dj_|chain_><sid>/<file>``:
    ``result.output`` is that file (``root / output`` keeps an absolute path
    as is), ``result.stream_url`` is ``/outputs/<dj_|chain_><sid>/<file>``
    served from the owner's directory. A candidate that resolves outside
    ``<root>/<something>/`` is refused.
    """
    result = job.get("result") or {}
    meta = job.get("meta") or {}
    owner = meta.get("user_id") or "local"
    candidates: list[Path] = []
    output = result.get("output")
    if output:
        candidates.append((root / str(output)).parent)
    stream_url = result.get("stream_url")
    if stream_url and str(stream_url).startswith("/outputs/"):
        candidates.append((root / owner / str(stream_url)[len("/outputs/"):]).parent)
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root) or len(resolved.relative_to(root).parts) < 2:
            logger.warning("admin takedown refused a path outside OUTPUTS_DIR",
                           extra={"job_id": job.get("job_id")})
            continue
        if resolved.is_dir():
            return resolved
    return None


@router.delete("/mix/{job_id}")
def admin_delete_mix(job_id: str):
    job = job_store.get_job(job_id)
    if job is None:
        # evicted from the in-memory cache: load the row and put it back so the
        # takedown below persists through the store's own write path
        job = job_store.get_job_from_db(job_id)
        if job is None:
            raise HTTPException(404, detail="mix not found")
        with job_store._lock:
            job = job_store._jobs.setdefault(job_id, job)

    if job.get("status") == JobStatus.RUNNING:
        job_store.cancel_job(job_id)

    root = Path(paths_mod.OUTPUTS_DIR).resolve()
    out = _mix_output_dir(job, root)
    if out is not None:
        shutil.rmtree(out, ignore_errors=True)

    updated = job_store.mark_job_takedown(job_id)
    if updated is not None:
        # the SSE broadcaster routes by meta.user_id, so this reaches the owner
        job_store._emit("job_failed", updated)
    return {
        "job_id": job_id,
        "removed_dir": out.relative_to(root).as_posix() if out else None,
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


@router.delete("/user/{user_id}/flag")
def admin_unflag_user(user_id: str):
    legal_store.unflag_user(user_id)
    return {"user_id": user_id, "flagged": False}
