"""
scripts/api/routers/spotify.py — Spotify integration endpoints.

Auth flow:
  GET  /spotify/status              — check if configured + user connected
  GET  /spotify/auth                — redirect to Spotify OAuth
  GET  /spotify/callback            — handle OAuth code exchange
  DELETE /spotify/disconnect        — remove stored user token

Catalogue (no login needed):
  GET  /spotify/search              — search Spotify catalogue
  GET  /spotify/track/{id}          — single track info + audio features
  POST /spotify/import              — download a Spotify track via yt-dlp

User library (requires login):
  GET  /spotify/playlists                    — user's playlists
  GET  /spotify/playlists/{id}/tracks        — tracks in a playlist
  GET  /spotify/top                          — user's top tracks
  POST /spotify/import-playlist              — queue entire playlist for download
"""

import re

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import RedirectResponse

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _check_job_cap
from scripts.api.schemas import JobResponse, JobType
from scripts.api.tasks import task_download

router = APIRouter()


def _spotify():
    """Lazy import — avoids crashing the whole API if Spotify is not configured."""
    from scripts.core.spotify import get_client
    return get_client()


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/spotify/status", tags=["spotify"])
def spotify_status():
    """Check whether Spotify is configured and whether the user is connected."""
    sp = _spotify()
    return {
        "configured": sp.is_configured(),
        "connected":  sp.is_connected(),
    }


# ── OAuth ─────────────────────────────────────────────────────────────────────

@router.get("/spotify/auth", tags=["spotify"])
def spotify_auth():
    """
    Redirect the browser to Spotify's authorization page.
    After the user logs in, Spotify redirects to /spotify/callback.
    """
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(
            status_code=400,
            detail="Spotify is not configured. Add client_id and client_secret to config.local.yaml.",
        )
    auth_url, _state = sp.get_auth_url()
    return RedirectResponse(url=auth_url)


@router.get("/spotify/callback", tags=["spotify"])
def spotify_callback(code: str = None, state: str = None, error: str = None):
    """
    OAuth callback — exchanges authorization code for access + refresh tokens.
    Redirects back to the Streamlit UI on success.
    """
    if error:
        return RedirectResponse(url=f"http://127.0.0.1:8501?spotify_error={error}")

    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter.")

    sp = _spotify()
    success = sp.exchange_code(code, state)
    if not success:
        return RedirectResponse(url="http://127.0.0.1:8501?spotify_error=token_exchange_failed")

    return RedirectResponse(url="http://127.0.0.1:8501?spotify_connected=1")


@router.delete("/spotify/disconnect", tags=["spotify"])
def spotify_disconnect():
    """Remove the stored user token (user stays Spotify-logged-in, just not linked here)."""
    _spotify().disconnect()
    return {"status": "disconnected"}


# ── Catalogue ─────────────────────────────────────────────────────────────────

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


@router.post("/spotify/import", tags=["spotify"])
def spotify_import(body: dict):
    """
    Download a Spotify track via yt-dlp.

    Accepts either a Spotify track dict (from /spotify/search) or just:
      { "query": "Artist - Track Name", "separate": false, "camelot": "8A", "bpm": 128.0 }

    Pre-populates the meta.json cache with Spotify's BPM, Camelot, and energy
    data so the recommendation engine has rich metadata immediately — no need
    to wait for a full librosa analysis.

    Uses the standard job_store.submit_job() path — fully tracked, rate-limited,
    and observable via GET /jobs/{job_id}.
    """
    query    = body.get("query", "").strip()
    separate = body.get("separate", False)
    camelot  = body.get("camelot", "")
    bpm      = body.get("bpm")
    energy   = body.get("energy")

    if not query:
        raise HTTPException(status_code=400, detail="'query' field is required.")

    _check_job_cap()

    job_id = job_store.create_job(JobType.DOWNLOAD, {"query": query, "source": "spotify"})
    job_store.submit_job(
        job_id, task_download,
        query=query,
        name=None,
        separate=separate,
    )

    # Pre-populate Spotify metadata into meta.json after the job finishes.
    # We do this in a lightweight fire-and-forget thread so it doesn't block
    # the job executor's worker slot and doesn't affect job lifecycle tracking.
    if bpm or camelot or energy is not None:
        import threading

        def _write_meta():
            try:
                from scripts.core.recommend import write_meta_cache
                from scripts.core.paths import LIBRARY_DIR
                # Wait briefly for the download job to complete before writing
                import time
                for _ in range(60):  # up to 60 s
                    job = job_store.get_job(job_id)
                    if job and job.get("status") in ("done", "failed"):
                        break
                    time.sleep(1)
                song_dir_path = LIBRARY_DIR / query
                if song_dir_path.exists():
                    write_meta_cache(
                        song_dir_path,
                        bpm=float(bpm) if bpm else 0.0,
                        camelot=camelot,
                        energy_mean=float(energy) if energy is not None else -1.0,
                    )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning("Could not write Spotify meta: %s", e)

        threading.Thread(target=_write_meta, daemon=True).start()

    return job_store.job_to_response(job_store.get_job(job_id))


# ── User library ──────────────────────────────────────────────────────────────

@router.get("/spotify/playlists", tags=["spotify"])
def spotify_playlists(limit: int = Query(50, le=50)):
    """Get the authenticated user's playlists. Requires user login."""
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")
    if not sp.is_connected():
        raise HTTPException(status_code=401, detail="User not connected. Visit /spotify/auth first.")

    playlists = sp.get_user_playlists(limit=limit)
    return {"playlists": [p.to_dict() for p in playlists], "count": len(playlists)}


@router.get("/spotify/playlists/{playlist_id}/tracks", tags=["spotify"])
def spotify_playlist_tracks(playlist_id: str, limit: int = Query(50, le=50)):
    """Get tracks in a specific playlist. Enriches with audio features if available."""
    # Spotify playlist IDs are base-62, 10–40 chars. Validate at boundary.
    if not re.fullmatch(r"[A-Za-z0-9]{10,40}", playlist_id):
        raise HTTPException(status_code=400, detail="Invalid playlist ID.")
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")

    tracks = sp.get_playlist_tracks(playlist_id, limit=limit)
    tracks = sp.enrich_tracks(tracks)
    return {"tracks": [t.to_dict() for t in tracks], "count": len(tracks)}


@router.get("/spotify/top", tags=["spotify"])
def spotify_top_tracks(
    limit: int      = Query(20, le=50),
    time_range: str = Query("medium_term", description="short_term | medium_term | long_term"),
):
    """Get the user's top tracks. Requires user login."""
    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")
    if not sp.is_connected():
        raise HTTPException(status_code=401, detail="User not connected. Visit /spotify/auth first.")

    tracks = sp.get_user_top_tracks(limit=limit, time_range=time_range)
    tracks = sp.enrich_tracks(tracks)
    return {"tracks": [t.to_dict() for t in tracks], "count": len(tracks)}


@router.post("/spotify/import-playlist", tags=["spotify"])
def spotify_import_playlist(body: dict):
    """
    Queue all tracks in a Spotify playlist for download.
    Returns a list of job_ids — one per track.

    Uses the standard job_store.submit_job() path — fully tracked, rate-limited,
    and observable via GET /jobs/{job_id}.

    Body: { "playlist_id": "...", "separate": false }
    """
    playlist_id = body.get("playlist_id", "").strip()
    separate    = body.get("separate", False)

    if not playlist_id:
        raise HTTPException(status_code=400, detail="'playlist_id' is required.")
    if not re.fullmatch(r"[A-Za-z0-9]{10,40}", playlist_id):
        raise HTTPException(status_code=400, detail="Invalid playlist ID format.")

    sp = _spotify()
    if not sp.is_configured():
        raise HTTPException(status_code=400, detail="Spotify not configured.")

    tracks = sp.get_playlist_tracks(playlist_id, limit=50)
    if not tracks:
        raise HTTPException(status_code=404, detail="No tracks found in playlist.")

    job_ids = []
    for track in tracks:
        _check_job_cap()
        query  = track.download_query
        job_id = job_store.create_job(
            JobType.DOWNLOAD, {"query": query, "source": "spotify_playlist", "playlist_id": playlist_id}
        )
        job_store.submit_job(
            job_id, task_download,
            query=query,
            name=None,
            separate=separate,
        )
        job_ids.append({"track": track.name, "query": query, "job_id": job_id})

    return {"queued": len(job_ids), "jobs": job_ids}
