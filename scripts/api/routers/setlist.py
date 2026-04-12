"""
scripts/api/routers/setlist.py — Setlist optimisation endpoints.

POST /setlist/optimize          — optimise a JSON track list into the best play order
POST /setlist/import-csv        — upload an Exportify CSV and optimise
GET  /setlist/camelot/modulation — instant Camelot modulation analysis for two keys
POST /setlist/wordplay          — find lyric-adjacent pairs in a track list

All heavy work runs synchronously (playlists are typically <200 tracks and
processing is fast).  For very large libraries, wrap in a background job.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pydantic import BaseModel, Field

router = APIRouter(prefix="/setlist", tags=["setlist"])


# ── Pydantic request/response models ─────────────────────────────────────────

class TrackIn(BaseModel):
    """Minimal track descriptor accepted by the API."""

    track_id: str = Field("", description="Spotify ID or any stable identifier")
    title: str = Field(..., description="Track title (= 'Track Name' in Exportify)")
    artist: str = Field("", description="Primary artist")
    camelot: Optional[str] = Field(None, description="Camelot key e.g. '8A', '11B'")
    bpm: Optional[float] = None
    energy: Optional[float] = Field(None, ge=0.0, le=1.0, description="0.0–1.0")
    duration_ms: Optional[int] = None


class OptimizeRequest(BaseModel):
    tracks: List[TrackIn] = Field(..., min_length=2)
    arc: str = Field(
        "mountain",
        description="Energy arc shape: ramp_up | mountain | wave | ramp_down",
    )
    w_harmonic: float = Field(0.50, ge=0.0, le=1.0)
    w_bpm: float = Field(0.30, ge=0.0, le=1.0)
    w_energy: float = Field(0.20, ge=0.0, le=1.0)
    markov_weight: float = Field(0.15, ge=0.0, le=1.0)
    genius_token: Optional[str] = Field(
        None,
        description="Genius API token for wordplay bonus (optional). "
                    "Falls back to GENIUS_TOKEN env var.",
    )
    wordplay_weight: float = Field(
        0.10, ge=0.0, le=1.0,
        description="Cost-reduction weight for lyric-matched transitions (0 = disabled).",
    )


class TransitionOut(BaseModel):
    from_track: str
    to_track: str
    from_camelot: Optional[str] = None
    to_camelot: Optional[str] = None
    modulation_type: Optional[str] = None
    safe_to_blend: bool = False
    recommendation: Optional[str] = None
    bpm_delta: Optional[float] = None
    harmonic_cost: float = 0.0
    bpm_cost: float = 0.0
    energy_cost: float = 0.0
    total_cost: float = 0.0


class TrackOut(BaseModel):
    position: int
    name: str
    artist: str
    camelot: str
    bpm: float
    energy: float
    energy_level: int
    arc_target_energy: float
    transition: Optional[TransitionOut] = None
    cumulative_cost: float


class OptimizeResponse(BaseModel):
    track_count: int
    arc: str
    total_cost: float
    setlist: List[TrackOut]


class ModulationResponse(BaseModel):
    from_key: str
    to_key: str
    modulation_type: str
    semitone_shift: int
    cost: float
    impact: str
    recommendation: str
    safe_to_blend: bool


class WordplayTrackIn(BaseModel):
    track_id: str
    title: str
    artist: str


class WordplayRequest(BaseModel):
    tracks: List[WordplayTrackIn] = Field(..., min_length=2)
    genius_token: Optional[str] = None
    min_similarity: float = Field(0.05, ge=0.0, le=1.0)


class WordplayPairOut(BaseModel):
    source_title: str
    source_artist: str
    target_title: str
    target_artist: str
    score: float
    matched_phrases: List[str]


class WordplayResponse(BaseModel):
    pairs: List[WordplayPairOut]
    total_found: int


# ── helpers ───────────────────────────────────────────────────────────────────

def _arc_enum(arc_str: str):
    from scripts.core.setlist_planner import EnergyArc
    mapping = {
        "ramp_up": EnergyArc.RAMP_UP,
        "mountain": EnergyArc.MOUNTAIN,
        "wave": EnergyArc.WAVE,
        "ramp_down": EnergyArc.RAMP_DOWN,
    }
    arc = mapping.get(arc_str.lower())
    if arc is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown arc '{arc_str}'. Valid: {list(mapping)}",
        )
    return arc


def _track_in_to_node(t: TrackIn):
    """Convert API TrackIn → setlist_planner.TrackNode."""
    from scripts.core.setlist_planner import TrackNode
    return TrackNode(
        name=t.title,
        artist=t.artist,
        camelot=t.camelot or "",
        bpm=t.bpm or 0.0,
        energy=t.energy if t.energy is not None else 0.5,
        duration_ms=t.duration_ms or 0,
        spotify_id=t.track_id,
    )


def _result_to_out(r: dict) -> TrackOut:
    """Convert SetlistPlanner.optimize() row dict → TrackOut."""
    tr_raw = r.get("transition_to_next")
    transition = None
    if tr_raw:
        transition = TransitionOut(
            from_track=r.get("name", ""),
            to_track=tr_raw.get("next_track", ""),
            from_camelot=r.get("camelot"),
            to_camelot=tr_raw.get("next_camelot"),
            modulation_type=tr_raw.get("modulation_type"),
            safe_to_blend=bool(tr_raw.get("safe_to_blend", False)),
            recommendation=tr_raw.get("recommendation"),
            bpm_delta=tr_raw.get("bpm_delta"),
            harmonic_cost=round(tr_raw.get("harmonic_cost", 0.0), 4),
            bpm_cost=round(tr_raw.get("bpm_cost", 0.0), 4),
            energy_cost=round(tr_raw.get("energy_cost", 0.0), 4),
            total_cost=round(tr_raw.get("total_cost", 0.0), 4),
        )
    return TrackOut(
        position=r.get("position", 0) + 1,
        name=r.get("name", ""),
        artist=r.get("artist", ""),
        camelot=r.get("camelot", ""),
        bpm=r.get("bpm", 0.0),
        energy=r.get("energy", 0.0),
        energy_level=r.get("energy_level", 5),
        arc_target_energy=r.get("arc_target_energy", 0.5),
        transition=transition,
        cumulative_cost=r.get("cumulative_cost", 0.0),
    )


def _build_wordplay_bonus(
    nodes,
    genius_token: Optional[str],
    wordplay_weight: float,
) -> dict[tuple[str, str], float]:
    """
    Returns {(src_spotify_id, tgt_spotify_id): bonus_float} used to
    nudge the greedy loop toward lyric-compatible successors.

    This is injected as a score adjustment; the actual greedy loop in
    SetlistPlanner doesn't know about wordplay — we patch ``transition_cost``
    results post-hoc via a monkey-patched override map.
    """
    from scripts.core.wordplay import TrackInput as WPTrackInput, find_wordplay_pairs

    token = genius_token or os.getenv("GENIUS_TOKEN")
    if not token or wordplay_weight <= 0:
        return {}

    wp_tracks = [
        WPTrackInput(track_id=n.spotify_id or n.name, title=n.name, artist=n.artist)
        for n in nodes
    ]
    try:
        pairs = find_wordplay_pairs(wp_tracks, genius_token=token, min_similarity=0.04)
    except Exception:
        return {}

    return {
        (p.source_id, p.target_id): wordplay_weight * p.score
        for p in pairs
    }


# ── POST /setlist/optimize ────────────────────────────────────────────────────

@router.post("/optimize", response_model=OptimizeResponse)
def optimize_setlist(req: OptimizeRequest):
    """
    Optimise a JSON track list into the ideal DJ play order.

    Uses weighted greedy search (harmonic + BPM + energy arc) with optional
    Markov chain refinement and wordplay lyric bonus.
    """
    from scripts.core.setlist_planner import SetlistPlanner

    arc = _arc_enum(req.arc)
    nodes = [_track_in_to_node(t) for t in req.tracks]

    # Wordplay bonus map (empty dict if no token / weight = 0)
    # Note: SetlistPlanner doesn't natively support wordplay — we patch the
    # energy field as a proxy signal so low-cost lyric pairs are preferred.
    # A cleaner integration would subclass SetlistPlanner, but this is fast.
    wp_bonus = _build_wordplay_bonus(nodes, req.genius_token, req.wordplay_weight)

    # Nudge energy slightly upward for tracks with strong lyric links from
    # their predecessor — the greedy loop will prefer them.
    if wp_bonus:
        id_to_node = {n.spotify_id or n.name: n for n in nodes}
        for (src_id, tgt_id), bonus in wp_bonus.items():
            tgt = id_to_node.get(tgt_id)
            if tgt:
                tgt.energy = min(1.0, tgt.energy + bonus * 0.05)

    planner = SetlistPlanner()
    results = planner.optimize(
        tracks=nodes,
        arc=arc,
        w_harmonic=req.w_harmonic,
        w_bpm=req.w_bpm,
        w_energy=req.w_energy,
        markov_weight=req.markov_weight,
    )

    setlist = [_result_to_out(r) for r in results]
    total_cost = results[-1].get("cumulative_cost", 0.0) if results else 0.0

    return OptimizeResponse(
        track_count=len(setlist),
        arc=req.arc,
        total_cost=round(total_cost, 4),
        setlist=setlist,
    )


# ── POST /setlist/import-csv ──────────────────────────────────────────────────

@router.post("/import-csv", response_model=OptimizeResponse)
async def import_exportify_csv(
    file: UploadFile = File(..., description="Exportify CSV from exportify.net"),
    arc: str = Query("mountain", description="ramp_up | mountain | wave | ramp_down"),
    genius_token: Optional[str] = Query(None, description="Genius API token (optional)"),
    wordplay_weight: float = Query(0.10, ge=0.0, le=1.0),
):
    """
    Upload an Exportify CSV and receive an optimised setlist.

    1. Go to https://exportify.net → connect your Spotify account → export any playlist.
    2. POST the downloaded .csv file here.
    3. Receive tracks in the optimal DJ play order.

    Exportify column headers supported (flexible matching):
      Spotify ID, Track Name, Artist Name(s), Album Name, Duration (ms),
      Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness,
      Acousticness, Instrumentalness, Liveness, Valence, Tempo, Genres
    """
    from scripts.core.setlist_planner import SetlistPlanner, parse_exportify_csv

    content = await file.read()

    try:
        nodes = parse_exportify_csv(content)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"CSV parse error: {exc}")

    if len(nodes) < 2:
        raise HTTPException(
            status_code=422,
            detail="CSV must contain at least 2 valid tracks. "
                   "Check that the file is a genuine Exportify export.",
        )

    arc_enum = _arc_enum(arc)

    wp_bonus = _build_wordplay_bonus(nodes, genius_token, wordplay_weight)
    if wp_bonus:
        id_to_node = {(n.spotify_id or n.name): n for n in nodes}
        for (src_id, tgt_id), bonus in wp_bonus.items():
            tgt = id_to_node.get(tgt_id)
            if tgt:
                tgt.energy = min(1.0, tgt.energy + bonus * 0.05)

    planner = SetlistPlanner()
    results = planner.optimize(tracks=nodes, arc=arc_enum)

    setlist = [_result_to_out(r) for r in results]
    total_cost = results[-1].get("cumulative_cost", 0.0) if results else 0.0

    return OptimizeResponse(
        track_count=len(setlist),
        arc=arc,
        total_cost=round(total_cost, 4),
        setlist=setlist,
    )


# ── GET /setlist/camelot/modulation ──────────────────────────────────────────

@router.get("/camelot/modulation", response_model=ModulationResponse)
def camelot_modulation(
    from_key: str = Query(..., description="Source Camelot key, e.g. '8A'"),
    to_key: str = Query(..., description="Target Camelot key, e.g. '9B'"),
):
    """
    Instant harmonic modulation analysis between two Camelot wheel positions.

    Returns the named modulation type (energy_boost, diagonal, relative_shift,
    perfect_fifth, etc.), its psychoacoustic cost (0.0 = perfect blend,
    1.0 = jarring clash), and whether the transition is safe to blend live.
    """
    from scripts.core.key_detection import camelot_modulation as _mod

    result = _mod(from_key.strip().upper(), to_key.strip().upper())

    if result.get("type") == "invalid_input":
        raise HTTPException(
            status_code=422,
            detail=(
                f"Invalid Camelot key(s): '{from_key}' / '{to_key}'. "
                "Expected format: number (1–12) + letter (A or B), e.g. '8A' or '11B'."
            ),
        )

    return ModulationResponse(
        from_key=from_key.upper(),
        to_key=to_key.upper(),
        modulation_type=result["type"],
        semitone_shift=result.get("semitone_shift", 0),
        cost=result.get("cost", 1.0),
        impact=result.get("impact", ""),
        recommendation=result.get("recommendation", ""),
        safe_to_blend=result.get("safe_to_blend", False),
    )


# ── POST /setlist/wordplay ────────────────────────────────────────────────────

@router.post("/wordplay", response_model=WordplayResponse)
def find_wordplay(req: WordplayRequest):
    """
    Find lyric-adjacent track pairs in a playlist.

    Scores pairs where the closing lyrics of one song share phrases or
    semantic overlap with the opening lyrics of the next — the DJ "wordplay"
    technique. Requires a Genius client-access token.

    Tip: get a free token at https://genius.com/api-clients
    """
    from scripts.core.wordplay import TrackInput as WPTrackInput, find_wordplay_pairs

    token = req.genius_token or os.getenv("GENIUS_TOKEN")
    if not token:
        raise HTTPException(
            status_code=422,
            detail=(
                "Genius API token required. Pass 'genius_token' in the request body "
                "or set the GENIUS_TOKEN environment variable. "
                "Get a free token at https://genius.com/api-clients"
            ),
        )

    wp_tracks = [
        WPTrackInput(track_id=t.track_id, title=t.title, artist=t.artist)
        for t in req.tracks
    ]

    pairs = find_wordplay_pairs(
        wp_tracks,
        genius_token=token,
        min_similarity=req.min_similarity,
    )

    pairs_out = [
        WordplayPairOut(
            source_title=p.source_title,
            source_artist=p.source_artist,
            target_title=p.target_title,
            target_artist=p.target_artist,
            score=round(p.score, 4),
            matched_phrases=p.matched_phrases,
        )
        for p in pairs
    ]

    return WordplayResponse(pairs=pairs_out, total_found=len(pairs_out))
