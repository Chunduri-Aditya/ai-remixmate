"""
scripts/api/routers/remix.py — DJ remix and beat synthesis endpoints.

POST /dj-remix                 — render a DJ transition (queued job)
POST /dj-chain                 — render an N-song continuous mix (queued job)
GET  /beat/synthesize          — synthesize a drum beat loop
POST /beat/upload              — upload a custom beat file
POST /instrument-lab           — stem swap experiments (queued job)
GET  /instrument-lab/songs     — list songs eligible for stem swap
"""

import re
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _require_song, _check_job_cap
from scripts.api.schemas import (
    DJChainRequest,
    DJPreviewRequest,
    DJRemixRequest,
    InstrumentLabRequest,
    JobResponse,
    JobType,
)
from scripts.api.tasks import (
    task_dj_chain,
    task_dj_remix,
    task_instrument_lab,
    task_remix_preview,
)
from scripts.core.paths import LIBRARY_DIR, OUTPUTS_DIR

router = APIRouter()


# ---------------------------------------------------------------------------
# DJ Remix (queued)
# ---------------------------------------------------------------------------

@router.post("/dj-remix", response_model=JobResponse, status_code=202, tags=["remix"])
def dj_remix(req: DJRemixRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song_a)
    _require_song(req.song_b)
    job_id = job_store.create_job(
        JobType.DJ_REMIX,
        {"song_a": req.song_a, "song_b": req.song_b},
    )
    job_store.submit_job(
        job_id, task_dj_remix,
        song_a=req.song_a,
        song_b=req.song_b,
        transition_bars=req.transition_bars,
        preset=req.preset,
        bridge_beat_mode=req.bridge_beat_mode,
        bridge_beat_genre=req.bridge_beat_genre,
        bridge_beat_intensity=req.bridge_beat_intensity,
        bridge_beat_path=req.bridge_beat_path,
        transition_effect=req.transition_effect,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# DJ Remix Preview (queued) — transition-only fast render for audition
# ---------------------------------------------------------------------------

@router.post("/dj-remix/preview", response_model=JobResponse, status_code=202, tags=["remix"])
def dj_remix_preview(req: DJPreviewRequest):
    """
    Render only the transition window between two songs for fast audition.

    Cheaper than a full dj-remix job — analyses both tracks, plans the
    transition, renders just the crossfade/effect region, and returns a
    streamable WAV URL alongside harmonic and tempo compatibility data.

    Result fields (available once the job is DONE):
      - stream_url       — relative URL to stream the preview WAV
      - duration_sec     — preview clip length in seconds
      - bpm_a / bpm_b    — detected tempos
      - harmonic_score   — 0–1 harmonic compatibility
      - camelot_a/b      — Camelot wheel keys
      - tempo_ratio      — bpm_b / bpm_a (1.0 = perfect match)
      - exit_bar_a       — bar in song A where the mix-out begins
      - entry_bar_b      — bar in song B where the mix-in ends
    """
    _check_job_cap()
    _require_song(req.song_a)
    _require_song(req.song_b)

    job_id = job_store.create_job(
        JobType.DJ_REMIX,
        {"song_a": req.song_a, "song_b": req.song_b, "preview": True},
    )
    job_store.submit_job(
        job_id,
        task_remix_preview,
        song_a=req.song_a,
        song_b=req.song_b,
        transition_bars=req.transition_bars,
        transition_effect=req.transition_effect,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# DJ Chain (queued) — N-song continuous mix
# ---------------------------------------------------------------------------

@router.post("/dj-chain", response_model=JobResponse, status_code=202, tags=["remix"])
def dj_chain(req: DJChainRequest, background_tasks: BackgroundTasks = None):
    """
    Render a continuous N-song DJ mix chain (2–8 songs).

    All songs are pre-warped to Song 1's BPM and stitched with phrase-locked
    transitions. Returns a single WAV containing the full chain mix.
    """
    _check_job_cap()
    for name in req.songs:
        _require_song(name)

    job_id = job_store.create_job(
        JobType.DJ_REMIX,
        {"songs": req.songs, "n": len(req.songs)},
    )
    job_store.submit_job(
        job_id, task_dj_chain,
        songs=req.songs,
        transition_bars=req.transition_bars,
        preset=req.preset,
        bridge_beat_mode=req.bridge_beat_mode,
        bridge_beat_genre=req.bridge_beat_genre,
        bridge_beat_intensity=req.bridge_beat_intensity,
        bridge_beat_path=req.bridge_beat_path,
        transition_effect=req.transition_effect,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# Beat synthesis & upload
# ---------------------------------------------------------------------------

@router.get("/beat/synthesize", tags=["beat"])
def synthesize_beat(
    bpm:       float = Query(128.0, ge=60, le=220, description="Target BPM"),
    genre:     str   = Query("auto",  description="techno|house|hiphop|trap|dnb|ambient|auto"),
    bars:      int   = Query(4,  ge=1, le=32, description="Beat loop length in bars"),
    intensity: float = Query(0.38, ge=0.0, le=1.0, description="Peak gain 0-1"),
):
    """
    Synthesize a drum beat loop using the built-in beat engine.

    Returns the beat audio as a WAV file *and* the matching Strudel code so
    the user can open strudel.cc and customise the pattern manually.
    """
    import soundfile as sf
    from scripts.core.beat_synth import render_beat, strudel_code

    audio  = render_beat(bpm=bpm, genre=genre, bars=bars, intensity=intensity)
    sr     = 22050

    beat_id  = str(uuid.uuid4())[:8]
    out_dir  = OUTPUTS_DIR / f"beat_{beat_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"bridge_{genre}_{int(bpm)}bpm_{bars}bars.wav"
    out_path = out_dir / filename
    sf.write(str(out_path), audio, sr)

    code = strudel_code(bpm=bpm, genre=genre)
    return {
        "beat_id":      beat_id,
        "audio_url":    f"/outputs/beat_{beat_id}/{filename}",
        "audio_path":   str(out_path),
        "bpm":          bpm,
        "genre":        genre,
        "bars":         bars,
        "strudel_code": code,
        "strudel_url":  "https://strudel.cc/",
    }


@router.post("/beat/upload", tags=["beat"])
async def upload_beat(file: UploadFile = File(...)):
    """
    Accept a user-recorded WAV (e.g. exported from Strudel) and store it
    so it can be referenced as a bridge beat in a DJ remix job.
    """
    beat_id  = str(uuid.uuid4())[:8]
    out_dir  = OUTPUTS_DIR / f"beat_{beat_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Validate file size (max 100 MB)
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")
    safe_name = re.sub(r"[^A-Za-z0-9_\-.]", "_", file.filename or "beat.wav")
    if not safe_name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".aiff")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    out_path  = out_dir / safe_name
    out_path.write_bytes(content)
    return {
        "beat_id":    beat_id,
        "audio_path": str(out_path),
        "filename":   safe_name,
        "size_kb":    round(len(content) / 1024, 1),
    }


# ---------------------------------------------------------------------------
# Instrument Lab — stem swap experiments
# ---------------------------------------------------------------------------

@router.post("/instrument-lab", response_model=JobResponse, status_code=202, tags=["lab"])
def instrument_lab(req: InstrumentLabRequest, background_tasks: BackgroundTasks = None):
    """
    Run the Instrument Lab — swap and mix stems between songs.

    Given 2+ songs with Demucs stems, generates all possible stem combinations
    (e.g. vocals from A + drums from B + bass from A + other from B) and renders
    each as a separate audio file.

    Modes:
      - "targeted" (default): single-stem swaps only (manageable count)
      - "all": full permutation (N^4 combos — can be large)

    Runs as an async job — poll /jobs/{job_id} for progress.
    """
    _check_job_cap()

    if len(req.songs) < 2:
        raise HTTPException(400, "Need at least 2 songs for instrument experiments")

    for song in req.songs:
        _require_song(song)

    job_id = job_store.create_job(JobType.DJ_REMIX, {
        "songs": req.songs, "type": "instrument_lab", "mode": req.mode,
    })
    job_store.submit_job(
        job_id,
        task_instrument_lab,
        songs=req.songs,
        mode=req.mode,
        swap_stems=req.swap_stems,
        target_duration=req.target_duration,
        include_pure=req.include_pure,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


@router.get("/instrument-lab/songs", tags=["lab"])
def instrument_lab_songs():
    """List all library songs that have Demucs stems (eligible for Instrument Lab)."""
    from scripts.core.instrument_lab import get_songs_with_stems
    songs = get_songs_with_stems(min_stems=2)
    return {"songs": songs, "count": len(songs)}
