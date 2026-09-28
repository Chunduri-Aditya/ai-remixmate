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
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _check_job_cap, _require_song
from scripts.api.schemas import (
    DJChainRequest,
    DJPreviewRequest,
    DJRemixRequest,
    InstrumentLabRequest,
    JobResponse,
    JobType,
    RemixFeedbackRequest,
    RemixFeedbackResponse,
)
from scripts.api.tasks import (
    task_dj_chain,
    task_dj_remix,
    task_instrument_lab,
    task_remix_preview,
)
from scripts.core.paths import user_outputs_dir

router = APIRouter()

_BEAT_EXTS = (".wav", ".flac")
_GENRE_RE = re.compile(r"^[a-z0-9_-]{1,32}$")
_BEAT_MAX_BYTES = 100 * 1024 * 1024


def _owns_session(session_id: str) -> bool:
    """True when the caller's outputs hold this session as a direct child:
    ``dj_<id>/`` for a full render, ``<id>/`` for a preview."""
    if not session_id or any(c in session_id for c in ("/", "\\", "\x00")):
        return False
    root = user_outputs_dir().resolve()
    for name in (f"dj_{session_id}", session_id):
        candidate = (root / name).resolve()
        if candidate.parent == root and candidate.is_dir():
            return True
    return False


def _confine_beat_path(raw: Optional[str]) -> Optional[str]:
    """Validate ``bridge_beat_path`` and return it relative to the user dir.

    Accepts a bare file name or a path that resolves inside
    ``user_outputs_dir()`` and ends in .wav or .flac; anything else is 400.
    The error never echoes the input or a server path.
    """
    if not raw:
        return None
    if "://" in raw or "\x00" in raw or ".." in Path(raw).parts:
        raise HTTPException(
            status_code=400, detail="bridge_beat_path must name a beat file in your outputs"
        )
    root = user_outputs_dir().resolve()
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or resolved.suffix.lower() not in _BEAT_EXTS:
        raise HTTPException(
            status_code=400,
            detail="bridge_beat_path must name a .wav or .flac file in your outputs",
        )
    return resolved.relative_to(root).as_posix()


# ---------------------------------------------------------------------------
# DJ Remix (queued)
# ---------------------------------------------------------------------------

@router.post("/dj-remix", response_model=JobResponse, status_code=202, tags=["remix"])
def dj_remix(req: DJRemixRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song_a)
    _require_song(req.song_b)
    bridge_beat_path = _confine_beat_path(req.bridge_beat_path)
    # meta.request is what the inspector's Retry replays; the beat path is the
    # confined relative form so no server path lands in a job response.
    request = {**req.model_dump(mode="json"), "bridge_beat_path": bridge_beat_path}
    job_id = job_store.create_job(
        JobType.DJ_REMIX,
        {"song_a": req.song_a, "song_b": req.song_b, "request": request},
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
        bridge_beat_path=bridge_beat_path,
        transition_effect=req.transition_effect,
    )
    return job_store.job_to_response(job_store.get_job(job_id))


# ---------------------------------------------------------------------------
# DJ Remix feedback — trains the spectral cue-matching band weights
# ---------------------------------------------------------------------------

@router.post("/dj-remix/feedback", response_model=RemixFeedbackResponse, tags=["remix"])
def dj_remix_feedback(req: RemixFeedbackRequest):
    """Record a 👍/👎 verdict on a rendered mix or preview.

    Synchronous — one deterministic weight update against the band
    similarities stashed at render time (see scripts/core/spectral_matching).
    404 when the session has no spectral record (matching disabled, chain
    render, or feedback already given — records are one-shot), and when the
    session directory is not under the caller's own outputs.
    """
    if not _owns_session(req.session_id):
        raise HTTPException(status_code=404, detail="session not found")
    from scripts.core.spectral_matching import apply_feedback
    summary = apply_feedback(req.session_id, liked=req.liked)
    if not summary.get("updated"):
        raise HTTPException(status_code=404, detail=summary.get("reason", "no record"))
    return RemixFeedbackResponse(**summary)


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
        {"song_a": req.song_a, "song_b": req.song_b, "preview": True,
         "request": req.model_dump(mode="json")},
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
    bridge_beat_path = _confine_beat_path(req.bridge_beat_path)

    request = {**req.model_dump(mode="json"), "bridge_beat_path": bridge_beat_path}
    job_id = job_store.create_job(
        JobType.DJ_REMIX,
        {"songs": req.songs, "n": len(req.songs), "request": request},
    )
    job_store.submit_job(
        job_id, task_dj_chain,
        songs=req.songs,
        transition_bars=req.transition_bars,
        preset=req.preset,
        bridge_beat_mode=req.bridge_beat_mode,
        bridge_beat_genre=req.bridge_beat_genre,
        bridge_beat_intensity=req.bridge_beat_intensity,
        bridge_beat_path=bridge_beat_path,
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
    if not _GENRE_RE.fullmatch(genre):
        raise HTTPException(status_code=400, detail="genre must match ^[a-z0-9_-]{1,32}$")

    import soundfile as sf

    from scripts.core.beat_synth import render_beat, strudel_code

    audio  = render_beat(bpm=bpm, genre=genre, bars=bars, intensity=intensity)
    sr     = 22050

    beat_id  = str(uuid.uuid4())[:8]
    out_dir  = user_outputs_dir() / f"beat_{beat_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"bridge_{genre}_{int(bpm)}bpm_{bars}bars.wav"
    out_path = out_dir / filename
    sf.write(str(out_path), audio, sr)

    code = strudel_code(bpm=bpm, genre=genre)
    return {
        "beat_id":      beat_id,
        "audio_url":    f"/outputs/beat_{beat_id}/{filename}",
        "audio_path":   f"beat_{beat_id}/{filename}",
        "bpm":          bpm,
        "genre":        genre,
        "bars":         bars,
        "strudel_code": code,
        "strudel_url":  "https://strudel.cc/",
    }


@router.post("/beat/upload", tags=["beat"])
async def upload_beat(request: Request):
    """
    Accept a user-recorded WAV (e.g. exported from Strudel) and store it
    so it can be referenced as a bridge beat in a DJ remix job.

    Multipart field ``file``. A Content-Length above 100 MB is 413 before any
    body is read; a body without one is read chunk by chunk and cut off at
    413 once the running total passes 100 MB.
    """
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > _BEAT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 100 MB)")

    received = 0

    async def _capped_receive():
        nonlocal received
        message = await request.receive()
        received += len(message.get("body", b""))
        if received > _BEAT_MAX_BYTES:
            raise HTTPException(status_code=413, detail="File too large (max 100 MB)")
        return message

    form = await Request(request.scope, _capped_receive).form()
    file = form.get("file")
    if file is None or isinstance(file, str):
        await form.close()
        raise HTTPException(status_code=422, detail="multipart field 'file' is required")
    content = await file.read()
    await form.close()
    safe_name = re.sub(r"[^A-Za-z0-9_\-.]", "_", file.filename or "beat.wav")
    if not safe_name.lower().endswith((".wav", ".flac", ".mp3", ".ogg", ".aiff")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    beat_id  = str(uuid.uuid4())[:8]
    out_dir  = user_outputs_dir() / f"beat_{beat_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path  = out_dir / safe_name
    out_path.write_bytes(content)
    return {
        "beat_id":    beat_id,
        "audio_path": f"beat_{beat_id}/{safe_name}",
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
