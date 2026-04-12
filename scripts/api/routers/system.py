"""
scripts/api/routers/system.py — System health and status endpoints.

GET  /health          — liveness check (library size + running jobs)
GET  /health/live     — lightweight liveness probe
GET  /health/ready    — readiness probe (thread pool + library access)
"""

from fastapi import APIRouter, HTTPException

from scripts.api import jobs as job_store
from scripts.core.paths import LIBRARY_DIR

router = APIRouter()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
def health():
    """Basic health check — library size + running jobs."""
    active_jobs = sum(1 for j in job_store.list_jobs() if j["status"] == "running")
    return {
        "status": "ok",
        "library_songs": sum(1 for d in LIBRARY_DIR.iterdir() if d.is_dir()),
        "active_jobs": active_jobs,
    }


@router.get("/health/live", tags=["system"])
def liveness():
    """Lightweight liveness probe for container orchestration."""
    return {"status": "ok"}


@router.get("/health/ready", tags=["system"])
def readiness():
    """Readiness probe — checks thread pool and library access."""
    from scripts.api.jobs import _executor

    # Check thread pool is functional
    try:
        if hasattr(_executor, '_broken') and _executor._broken:
            raise HTTPException(503, detail="Thread pool broken")
    except AttributeError:
        pass

    # Check library directory accessible
    if not LIBRARY_DIR.exists():
        raise HTTPException(503, detail="Library directory not found")

    return {"status": "ready"}
