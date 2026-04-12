"""
scripts/api/routers/library.py — Song library management endpoints.

GET  /library                              — list all songs + stats
GET  /library/names                        — lightweight song name list
GET  /library/{name}                       — single song detail
DELETE /library/{name}                     — remove song from library
GET  /library/{name}/audio                 — stream the full.wav
GET  /library/{name}/stems/{stem}          — stream a specific stem
GET  /outputs/{session_id}/{filename}      — stream a rendered mix
POST /library/initialize                   — one-shot pipeline
"""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _require_song, _song_info, _stem_file, _check_job_cap
from scripts.api.schemas import (
    InitializeLibraryRequest,
    JobResponse,
    JobType,
    LibraryListResponse,
    LibraryStats,
    SongInfo,
)
from scripts.api.tasks import task_initialize_library
from scripts.core.paths import LIBRARY_DIR, OUTPUTS_DIR

router = APIRouter()


@router.get("/library", response_model=LibraryListResponse, tags=["library"])
def list_library(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=5000),
    search: Optional[str] = Query(None, description="Filter by name (case-insensitive substring)"),
):
    all_dirs = sorted(
        (d for d in LIBRARY_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.name.lower(),
    )
    if search:
        pat = search.lower()
        all_dirs = [d for d in all_dirs if pat in d.name.lower()]

    total = len(all_dirs)
    page_dirs = all_dirs[(page - 1) * per_page : page * per_page]
    songs = [_song_info(d) for d in page_dirs]

    total_bytes = sum(
        f.stat().st_size
        for d in LIBRARY_DIR.iterdir() if d.is_dir()
        for f in d.rglob("*") if f.is_file()
    )
    total_gb = total_bytes / 1_073_741_824

    try:
        from scripts.core.config import cfg
        cap_gb = cfg.library.max_size_gb
    except Exception:
        cap_gb = 50.0

    songs_with_stems = sum(
        1 for d in LIBRARY_DIR.iterdir()
        if d.is_dir() and _stem_file(d, "vocals") is not None
    )

    stats = LibraryStats(
        total_songs=total,
        total_size_gb=round(total_gb, 2),
        cap_gb=cap_gb,
        within_cap=total_gb <= cap_gb,
        songs_with_stems=songs_with_stems,
    )
    return LibraryListResponse(stats=stats, songs=songs)


@router.get("/library/names", tags=["library"])
def list_library_names(
    with_wav: bool = Query(False, description="Only songs that have full.wav"),
    with_stems: bool = Query(False, description="Only songs that have Demucs stems"),
):
    """
    Return ALL song names in the library — lightweight, no pagination.
    Use this instead of /library?per_page=... when you only need the names.
    """
    all_dirs = sorted(
        (d for d in LIBRARY_DIR.iterdir() if d.is_dir()),
        key=lambda d: d.name.lower(),
    )
    names = []
    for d in all_dirs:
        if with_wav and not (d / "full.wav").exists():
            continue
        if with_stems and not any(
            (d / f"{s}.wav").exists() or (d / f"{s}.flac").exists()
            for s in ("vocals", "drums", "bass", "other")
        ):
            continue
        names.append(d.name)
    return {"names": names, "count": len(names)}


@router.get("/library/{name}", response_model=SongInfo, tags=["library"])
def get_song(name: str):
    return _song_info(_require_song(name))


@router.delete("/library/{name}", tags=["library"])
def delete_song(name: str):
    import shutil
    d = _require_song(name)
    shutil.rmtree(d)
    return {"deleted": name}


@router.get("/library/{name}/audio", tags=["library"])
def stream_audio(name: str):
    d = _require_song(name)
    wav = d / "full.wav"
    if not wav.exists():
        raise HTTPException(status_code=404, detail=f"full.wav not found for {name!r}")
    return FileResponse(str(wav), media_type="audio/wav", filename=f"{name}.wav")


@router.get("/library/{name}/stems/{stem}", tags=["library"])
def stream_stem(name: str, stem: str):
    if stem not in ("vocals", "drums", "bass", "other"):
        raise HTTPException(status_code=400, detail="stem must be one of: vocals drums bass other")
    d = _require_song(name)
    f = _stem_file(d, stem)
    if f is None:
        raise HTTPException(status_code=404, detail=f"{stem} stem not found for {name!r}")
    media_type = "audio/flac" if f.suffix == ".flac" else "audio/wav"
    return FileResponse(str(f), media_type=media_type, filename=f"{name}_{stem}{f.suffix}")


# ---------------------------------------------------------------------------
# Outputs (rendered mixes)
# ---------------------------------------------------------------------------

@router.get("/outputs/{session_id}/{filename}", tags=["outputs"])
def stream_output(session_id: str, filename: str):
    # Safety: reject path traversal attempts
    if ".." in session_id or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
    path = OUTPUTS_DIR / session_id / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output not found")
    return FileResponse(str(path), media_type="audio/wav", filename=filename)


# ---------------------------------------------------------------------------
# Full library initialisation (stem split → compress → index, one job)
# ---------------------------------------------------------------------------

@router.post("/library/initialize", response_model=JobResponse, status_code=202, tags=["library"])
def initialize_library(req: InitializeLibraryRequest, background_tasks=None):
    """
    One-shot pipeline: stem-split all unsplit songs → FLAC compress → rebuild RAG index.
    Returns a single job_id so the frontend can track all three phases from one progress bar.
    """
    _check_job_cap()
    job_id = job_store.create_job(
        JobType.ANALYZE, {"type": "initialize_library"}
    )
    job_store.submit_job(
        job_id, task_initialize_library,
        enhance=req.enhance,
        model=req.model,
        delete_wav=req.delete_wav,
        run_compress=req.run_compress,
        run_index=req.run_index,
    )
    return job_store.job_to_response(job_store.get_job(job_id))
