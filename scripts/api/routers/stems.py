"""
scripts/api/routers/stems.py — Stem separation and compression endpoints.

POST /stems/split              — stem-split a single song (queued job)
POST /stems/split-batch        — stem-split multiple songs (queued job)
POST /stems/compress           — compress stems for a single song (queued job)
POST /stems/compress-batch     — compress stems for all library songs (queued job)

The two split routes run Demucs (CPU, 2–5x realtime, one worker) and are the
only request-path trigger for it — see app/DEPLOYMENT_PLAN.md Phase 3. They are
gated per-route behind REMIXMATE_STEM_SPLIT_ENABLED (on locally, off in prod);
the two compress routes are FLAC encode only and stay open.
"""

import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query

from scripts.api import jobs as job_store
from scripts.api.auth import require_operator
from scripts.api.routers._helpers import _check_job_cap, _require_song
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


def _stem_split_enabled() -> None:
    """503 unless REMIXMATE_STEM_SPLIT_ENABLED is truthy. Read per request so a
    test (or an operator) can flip it without a restart. Guards only the Demucs
    split routes — an unauthenticated caller must not be able to pin the
    single-worker box on CPU stem separation (app/DEPLOYMENT_PLAN.md Phase 3)."""
    if os.environ.get("REMIXMATE_STEM_SPLIT_ENABLED", "").strip().lower() not in ("1", "true", "yes"):
        raise HTTPException(
            status_code=503,
            detail="Stem separation is disabled on this deployment.",
        )


@router.post(
    "/stems/split",
    response_model=JobResponse,
    status_code=202,
    tags=["stems"],
    dependencies=[Depends(require_operator), Depends(_stem_split_enabled)],
)
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


@router.post(
    "/stems/split-batch",
    response_model=JobResponse,
    status_code=202,
    tags=["stems"],
    dependencies=[Depends(require_operator), Depends(_stem_split_enabled)],
)
def stem_split_batch(req: BatchStemSplitRequest, background_tasks: BackgroundTasks = None):
    """
    Run Demucs on a list of songs (or ALL library songs if songs=null).
    Songs that already have all 4 stems are skipped unless skip_existing=false.
    """
    _check_job_cap()
    songs = req.songs
    if not songs:
        # Default: every library song separate_song_stems() can actually
        # consume — full.wav or its full_enhanced.wav fallback (see
        # scripts/core/audio_source.py::resolve_source_file()). Deliberately
        # NOT has_source_audio(), which also returns True for stems-only
        # songs (already fully split, or split with full.wav/full_enhanced
        # pruned away) — those have no single-file source for
        # separate_song_stems() to read, so including them here just
        # enqueues a song that immediately fails with "no source audio".
        from scripts.core.audio_source import resolve_source_file
        songs = [
            d.name for d in sorted(LIBRARY_DIR.iterdir())
            if d.is_dir() and resolve_source_file(d.name) is not None
        ]
    else:
        # req.songs is caller-supplied — validate each name the same way
        # _require_song() does (path-traversal confinement to LIBRARY_DIR)
        # before it can reach _prune_after_split()'s unlink() calls.
        # Invalid/nonexistent entries are dropped rather than failing the
        # whole batch, matching delete_songs_batch's best-effort pattern.
        validated = []
        for name in songs:
            try:
                _require_song(name)
                validated.append(name)
            except HTTPException:
                continue
        songs = validated
    if not songs:
        raise HTTPException(status_code=400, detail="No songs found in library with usable source audio")

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


@router.post(
    "/stems/compress",
    response_model=JobResponse,
    status_code=202,
    tags=["stems"],
    dependencies=[Depends(require_operator)],
)
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


@router.post(
    "/stems/compress-batch",
    response_model=JobResponse,
    status_code=202,
    tags=["stems"],
    dependencies=[Depends(require_operator)],
)
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
