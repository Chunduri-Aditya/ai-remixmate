"""
scripts/api/routers/downloads.py — Track download endpoints.

POST /download              — download a single track (queued job)
POST /download-playlist     — download all tracks from a playlist (queued job)
"""

from fastapi import APIRouter, BackgroundTasks

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _check_job_cap
from scripts.api.schemas import (
    DownloadRequest,
    JobResponse,
    JobType,
    PlaylistDownloadRequest,
)
from scripts.api.tasks import task_download, task_playlist_download

router = APIRouter()


@router.post("/download", response_model=JobResponse, status_code=202, tags=["downloads"])
def download_song(req: DownloadRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    job_id = job_store.create_job(JobType.DOWNLOAD, {"query": req.query})
    job_store.submit_job(
        job_id, task_download,
        query=req.query,
        name=req.name,
        separate=req.separate,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


@router.post("/download-playlist", response_model=JobResponse, status_code=202, tags=["downloads"])
def download_playlist_route(req: PlaylistDownloadRequest, background_tasks: BackgroundTasks = None):
    """
    Download all tracks from a YouTube / YouTube Music playlist URL.
    Returns a job ID — poll /jobs/{job_id} for per-track progress.
    """
    _check_job_cap()
    job_id = job_store.create_job(JobType.DOWNLOAD, {"url": req.url, "type": "playlist"})
    job_store.submit_job(
        job_id, task_playlist_download,
        url=req.url,
        separate=req.separate,
        limit=req.limit,
    )
    return job_store.job_to_response(job_store.get_job(job_id))
