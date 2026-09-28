"""
scripts/api/routers/feedback.py — Behavioral telemetry endpoints
(SELF_IMPROVING_DJ_RESEARCH.md §3, Stage A).

POST /feedback/event      — log one playback/UI event (fire-and-forget from the frontend)
GET  /feedback/summary    — aggregate counts per song / song-pair (Stage B offline analysis)

This is instrumentation only — see scripts/core/feedback_store.py's module
docstring. Nothing here scores or ranks anything; it collects the event
stream Stage B will later analyze to decide whether Stage C (a learned
correction term) is even worth building.
"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from scripts.api import jobs as job_store
from scripts.api.auth import get_user_id
from scripts.api.schemas import FeedbackEventRequest, FeedbackEventResponse, FeedbackSummaryResponse
from scripts.core import feedback_store
from scripts.core.logging_utils import get_logger

_log = get_logger(__name__)

router = APIRouter()


def _context_from_job(job_id: str) -> Dict[str, Any]:
    """
    Backfill `context` from a job's stored meta/result — SELF_IMPROVING_DJ_
    RESEARCH.md §3's context column needs "transition_bars, preset, effect,
    compatibility scores AT THE TIME of the render", and those already live
    on the job record (meta carries request params, result carries what the
    render actually computed) rather than requiring every caller to
    reassemble them. Result values win on overlap since they reflect what
    actually rendered, not just what was requested.

    Returns {} (not an error) for an unknown job_id — a stale/expired job_id
    on an old client shouldn't block logging the event itself.
    """
    job = job_store.get_job(job_id)
    if job is None or job_store._job_user(job) != get_user_id():
        return {}
    meta = job.get("meta") or {}
    result = job.get("result") or {}
    keys = (
        "transition_bars", "preset", "effect",
        "harmonic_score", "transition_confidence", "tempo_shift_ratio",
    )
    context: Dict[str, Any] = {}
    for k in keys:
        if k in meta:
            context[k] = meta[k]
        if k in result:
            context[k] = result[k]
    return context


@router.post("/feedback/event", response_model=FeedbackEventResponse, tags=["feedback"])
def log_feedback_event(req: FeedbackEventRequest):
    context = req.context
    if context is None and req.job_id:
        context = _context_from_job(req.job_id) or None

    try:
        event_id = feedback_store.log_event(
            event_type=req.event_type.value,
            song_a=req.song_a,
            song_b=req.song_b,
            job_id=req.job_id,
            position_sec=req.position_sec,
            context=context,
        )
    except ValueError as exc:
        # log_event() only raises for a genuinely invalid event_type — in
        # practice FastAPI's own enum validation on FeedbackEventType catches
        # that before we get here, so this is a defensive fallback, not the
        # primary validation path.
        raise HTTPException(status_code=400, detail=str(exc))

    return FeedbackEventResponse(event_id=event_id, logged=True)


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse, tags=["feedback"])
def feedback_summary():
    return FeedbackSummaryResponse(**feedback_store.summarize())
