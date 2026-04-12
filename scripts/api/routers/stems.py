"""
scripts/api/routers/stems.py — Stem separation and compression endpoints.

POST /stems/split              — stem-split a single song (queued job)
POST /stems/split-batch        — stem-split multiple songs (queued job)
POST /stems/compress           — compress stems for a single song (queued job)
POST /stems/compress-batch     — compress stems for all library songs (queued job)
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _require_song, _check_job_cap
from scripts.api.schemas import (
    BatchStemSplitRequest,
    JobResponse,
    JobType,
    StemSplitRequest,
)
from scripts.api.tasks import (
    task_batch_compress_stems,
    task_batch_stem_split,
    task_compress_stems,
    task_stem_split,
)
from scripts.core.paths import LIBRARY_DIR

router = APIRouter()


@router.post("/stems/split", response_model=JobResponse, status_code=202, tags=["stems"])
def stem_split(req: StemSplitRequest, background_tasks: BackgroundTasks = None):
    """
    Enhance a single library song then run Demucs to produce
    vocals / drums / bass / other stems.  Takes 2–5 minutes per song.
    """
    _check_job_cap()
    _require_song(req.song)
    job_id = job_store.create_job(JobType.ANALYZE, {"song": req.song, "type": "stem_split"})
    job_store.submit_job(
        job_id, task_stem_split,
        song=req.song,
        enhance=req.enhance,
        model=req.model,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


@router.post("/stems/split-batch", response_model=JobResponse, status_code=202, tags=["stems"])
def stem_split_batch(req: BatchStemSplitRequest, background_tasks: BackgroundTasks = None):
    """
    Run Demucs on a list of songs (or ALL library songs if songs=null).
    Songs that already have vocals.wav are skipped unless skip_existing=false.
    """
    _check_job_cap()
    songs = req.songs
    if not songs:
        # Default: all library songs that have a full.wav
        songs = [
            d.name for d in sorted(LIBRARY_DIR.iterdir())
            if d.is_dir() and (d / "full.wav").exists()
        ]
    if not songs:
        raise HTTPException(status_code=400, detail="No songs found in library with full.wav")

    job_id = job_store.create_job(
        JobType.ANALYZE, {"n_songs": len(songs), "type": "batch_stem_split"}
    )
    job_store.submit_job(
        job_id, task_batch_stem_split,
        songs=songs,
        enhance=req.enhance,
        model=req.model,
        skip_existing=req.skip_existing,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


@router.post("/stems/compress", response_model=JobResponse, status_code=202, tags=["stems"])
def compress_stems_single(
    song: str = Query(..., description="Song name in library"),
    delete_wav: bool = Query(True, description="Delete WAV after FLAC encode"),
    background_tasks: BackgroundTasks = None,
):
    """
    Compress WAV stems → lossless FLAC (~50% space, same quality).
    Runs as an async job.
    """
    _check_job_cap()
    _require_song(song)
    job_id = job_store.create_job(JobType.ANALYZE, {"song": song, "type": "compress_stems"})
    job_store.submit_job(job_id, task_compress_stems, song=song, delete_wav=delete_wav)
    return job_store.job_to_response(job_store.get_job(job_id))


@router.post("/stems/compress-batch", response_model=JobResponse, status_code=202, tags=["stems"])
def compress_stems_batch(
    background_tasks: BackgroundTasks = None,
    delete_wav: bool = Query(True, description="Delete WAVs after FLAC encode"),
    skip_existing: bool = Query(True, description="Skip songs that already have FLAC stems"),
):
    """
    Batch-compress all library stem WAVs to lossless FLAC.
    Pass songs=null to process the entire library.
    Poll /jobs/{job_id} for per-song progress.
    """
    _check_job_cap()
    job_id = job_store.create_job(
        JobType.ANALYZE, {"type": "batch_compress_stems"}
    )
    job_store.submit_job(
        job_id, task_batch_compress_stems,
        delete_wav=delete_wav,
        skip_existing=skip_existing,
    )
    return job_store.job_to_response(job_store.get_job(job_id))
