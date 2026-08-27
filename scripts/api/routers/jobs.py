"""
scripts/api/routers/jobs.py — Job queue management endpoints.

GET    /jobs             — list recent jobs (newest first)
GET    /jobs/{job_id}    — poll a specific job
DELETE /jobs/{job_id}    — cancel a pending or running job
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from scripts.api import jobs as job_store
from scripts.api.schemas import JobResponse

router = APIRouter()


@router.get("/jobs", response_model=List[JobResponse], tags=["jobs"])
def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    job_type: Optional[str] = Query(None, description="Filter by job type, e.g. 'dj_remix'"),
    status: Optional[str] = Query(None, description="Filter by status, e.g. 'done'"),
):
    """
    Return the most recent jobs, newest first.

    Filters apply before the limit, so ``?job_type=dj_remix&status=done`` returns the
    newest completed remixes rather than the completed remixes that happen to fall in
    the newest ``limit`` jobs overall.
    """
    return [
        job_store.job_to_response(j)
        for j in job_store.list_jobs(limit, job_type=job_type, status=status)
    ]


@router.get("/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
def get_job(job_id: str):
    """Poll the status, progress, ETA, and result of a specific job."""
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job_store.job_to_response(job)


@router.delete("/jobs/{job_id}", tags=["jobs"])
def cancel_job(job_id: str):
    """
    Request cancellation of a job.

    - PENDING jobs are cancelled immediately.
    - RUNNING jobs have their status set to cancelled; the task function is
      responsible for checking job status and exiting early.
    - Returns 404 if the job does not exist.
    - Returns 409 if the job is already complete.
    """
    job = job_store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    cancelled = job_store.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is in state '{job['status']}' and cannot be cancelled",
        )

    return JSONResponse({"job_id": job_id, "status": "cancelled"})
