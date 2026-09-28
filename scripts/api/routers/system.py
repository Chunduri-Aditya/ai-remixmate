"""
scripts/api/routers/system.py — System health and status endpoints.

GET  /health          — liveness check (library size + running jobs)
GET  /health/live     — lightweight liveness probe
GET  /health/ready    — readiness probe (thread pool, library, store DBs, catalog)
"""

import os

from fastapi import APIRouter, HTTPException
from sqlalchemy.exc import SQLAlchemyError

from scripts.api import jobs as job_store
from scripts.core import crates, db, feedback_store, legal_store
from scripts.core.logging_utils import get_logger
from scripts.core.paths import LIBRARY_DIR

router = APIRouter()
logger = get_logger(__name__)

# Store name -> module owning that store's ``_DB_PATH`` (read at call time so
# test path overrides and DATABASE_URL both apply).
_STORES = {
    "jobs": job_store,
    "crates": crates,
    "feedback": feedback_store,
    "legal": legal_store,
}


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", tags=["system"])
def health():
    """Basic health check — library size + running jobs."""
    active_jobs = sum(
        1 for j in job_store.list_jobs(user_id="*") if j["status"] == "running"
    )
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
    """Readiness probe — checks thread pool, library access, each store engine
    (``SELECT 1``) and the catalog size. ``REMIXMATE_REQUIRE_CATALOG=1`` makes an
    empty catalog a 503 with ``reason: catalog_empty``."""
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

    # Each store's database answers SELECT 1. The error text stays in the log:
    # it can carry a server path, which a response must not.
    for name, store in _STORES.items():
        try:
            db.ping(db.get_engine(str(store._DB_PATH)))
        except SQLAlchemyError as exc:
            logger.warning(f"readiness: store {name} unavailable: {exc}")
            raise HTTPException(
                503, detail={"reason": "store_unavailable", "store": name}
            ) from None

    catalog_songs = sum(1 for d in LIBRARY_DIR.iterdir() if d.is_dir())
    if catalog_songs == 0 and os.environ.get("REMIXMATE_REQUIRE_CATALOG", "").strip() == "1":
        raise HTTPException(503, detail={"reason": "catalog_empty", "catalog_songs": 0})

    return {"status": "ready", "catalog_songs": catalog_songs}
