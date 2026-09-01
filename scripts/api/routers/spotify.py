"""
scripts/api/routers/spotify.py — Spotify metadata endpoints (read-only).

The hosted product's audio comes from the operator-curated CC catalogue, not
Spotify. Spotify is kept only as an optional BPM/key/energy enrichment source,
using app credentials — no user OAuth, no import.

  GET  /spotify/status            — is Spotify configured?
  GET  /spotify/search            — search the Spotify catalogue (metadata only)
  GET  /spotify/track/{id}        — single track info + audio features

The OAuth / playlist / top-tracks / import endpoints were removed with the
catalogue teardown (deployment plan, Phase 1).
"""

import re

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()


def _spotify():
    """Lazy import — avoids crashing the whole API if Spotify is not configured."""
    from scripts.core.spotify import get_client
    return get_client()


@router.get("/spotify/status", tags=["spotify"])
def spotify_status():
    """Check whether Spotify metadata enrichment is configured."""
    sp = _spotify()
    return {"configured": sp.is_configured()}


@router.get("/spotify/search", tags=["spotify"])
def spotify_search(q: str = Query(..., description="Search query"), limit: int = Query(10, le=50)):
    """
    Search Spotify's catalogue. Returns tracks with optional audio features
    (BPM, key, energy, danceability) if the endpoint is available for your app.
    """
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")

    tracks = sp.search(q, limit=limit)
    tracks = sp.enrich_tracks(tracks)   # attempts audio features, gracefully skips on failure
    return {"tracks": [t.to_dict() for t in tracks], "query": q, "count": len(tracks)}


@router.get("/spotify/track/{track_id}", tags=["spotify"])
def spotify_track(track_id: str):
    """Get full info + audio features for a single Spotify track."""
    # Validate at the boundary: Spotify IDs are base-62, 10–30 chars.
    # Explicit re.fullmatch here terminates CodeQL's taint flow before
    # track_id reaches any HTTP call — prevents partial SSRF flagging.
    if not re.fullmatch(r"[A-Za-z0-9]{10,30}", track_id):
        raise HTTPException(status_code=400, detail="Invalid track ID.")
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")

    tracks = sp.search(f"id:{track_id}", limit=1)
    if not tracks:
        # Fallback: just get audio features
        features = sp.audio_features(track_id)
        return {"track": None, "audio_features": features}

    track = tracks[0]
    features = sp.audio_features(track_id)
    if features:
        track.bpm          = features.get("bpm")
        track.key          = features.get("key")
        track.mode         = features.get("mode")
        track.energy       = features.get("energy")
        track.danceability = features.get("danceability")
        track.valence      = features.get("valence")
        track.loudness_db  = features.get("loudness_db")
        track.camelot      = features.get("camelot")
        track.key_name     = features.get("key_name")

    return track.to_dict()
