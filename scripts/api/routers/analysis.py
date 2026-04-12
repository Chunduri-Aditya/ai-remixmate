"""
scripts/api/routers/analysis.py — Music analysis and recommendation endpoints.

POST /compatibility                    — instant Camelot + BPM check
POST /analyze                          — genre + structure analysis (queued job)
GET  /recommend/{name}                 — top compatible songs for a track
GET  /library/similar/{name}           — RAG vector similarity search
GET  /index/stats                      — RAG index statistics
POST /index/rebuild                    — rebuild RAG index (queued job)
"""

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query

from scripts.api import jobs as job_store
from scripts.api.routers._helpers import _require_song, _check_job_cap
from scripts.api.schemas import (
    AnalyzeRequest,
    CompatibilityRequest,
    CompatibilityResult,
    JobResponse,
    JobType,
)
from scripts.api.tasks import task_analyze, task_rebuild_index
from scripts.core.paths import LIBRARY_DIR

router = APIRouter()


@router.post("/compatibility", response_model=CompatibilityResult, tags=["analysis"])
def check_compatibility(req: CompatibilityRequest):
    """
    Instant Camelot + BPM compatibility check.
    Uses local audio analysis if songs are in the library,
    otherwise falls back to metadata API lookup.
    """
    import librosa
    from scripts.core.track_metadata import TrackMetadata, MetadataClient
    from scripts.core.dj_engine import _analyze_impl
    from scripts.core.genre import detect_genre

    client = MetadataClient()
    sr = 22050

    def _meta_for(name: str, artist: str) -> tuple[TrackMetadata, Optional[str]]:
        d = LIBRARY_DIR / name
        if d.exists() and (d / "full.wav").exists():
            audio, _ = librosa.load(str(d / "full.wav"), sr=sr, mono=True, duration=60.0)
            struct = _analyze_impl(audio, sr)
            genre_r = detect_genre(audio, sr)
            meta = TrackMetadata(
                title=name,
                artist=artist,
                bpm=struct.bpm,
                genres=[genre_r.genre],
            )
            return meta, genre_r.genre
        # Fall back to API
        meta = client.lookup(name, artist=artist)
        return meta, (meta.genres[0] if meta.genres else None)

    meta_a, genre_a = _meta_for(req.song_a, req.artist_a)
    meta_b, genre_b = _meta_for(req.song_b, req.artist_b)
    score = client.compatibility_score(meta_a, meta_b)

    return CompatibilityResult(
        song_a=req.song_a,
        song_b=req.song_b,
        compatible=score.get("compatible", False),
        overall=round(score.get("overall", 0.0), 3),
        bpm_score=round(score.get("bpm_score", 0.0), 3),
        key_score=round(score.get("key_score", 0.0), 3),
        energy_score=round(score.get("energy_score", 0.0), 3),
        bpm_a=round(meta_a.bpm, 1),
        bpm_b=round(meta_b.bpm, 1),
        camelot_a=meta_a.camelot,
        camelot_b=meta_b.camelot,
        genre_a=genre_a,
        genre_b=genre_b,
    )


@router.post("/analyze", response_model=JobResponse, status_code=202, tags=["analysis"])
def analyze_song(req: AnalyzeRequest, background_tasks: BackgroundTasks = None):
    _check_job_cap()
    _require_song(req.song)
    job_id = job_store.create_job(JobType.ANALYZE, {"song": req.song})
    job_store.submit_job(job_id, task_analyze, song=req.song)
    return job_store.job_to_response(job_store.get_job(job_id))


@router.get("/recommend/{name}", tags=["analysis"])
def recommend_songs(
    name: str,
    limit: int = Query(5, ge=1, le=20),
):
    """
    Return the top compatible songs for a given library track.

    Uses a per-song ``meta.json`` BPM cache for instant lookups.
    On first call the cache is populated lazily (up to 120 uncached songs),
    which may take 20–30 s; subsequent calls are typically < 1 s.
    """
    from scripts.core.recommend import get_recommendations

    src = _require_song(name)
    recommendations = get_recommendations(src, LIBRARY_DIR, limit=limit)
    return {"song": name, "recommendations": recommendations}


@router.get("/library/similar/{name}", tags=["index"])
def similar_songs(
    name: str,
    k: int = Query(5, ge=1, le=20, description="Number of results"),
):
    """
    Return the top-k most similar songs using the RAG vector index.

    Similarity is weighted cosine distance over a 35-dim feature vector:
    BPM (40%), key/mode (35%), energy (10%), rhythm (8%), vocal/timbre (7%).

    Much richer than pure BPM matching — takes key, energy, groove, and
    vocal density into account simultaneously.
    """
    _require_song(name)
    try:
        from scripts.core.music_index import get_index
        idx = get_index()
        # Upsert if not yet indexed
        if name not in idx._songs:
            idx.upsert_song(name)
        results = idx.search(name, k=k)
        if not results:
            # Cold start: try to upsert and retry
            idx.upsert_song(name)
            results = idx.search(name, k=k)
        return {"source": name, "similar": results, "engine": "rag_vector"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/index/stats", tags=["index"])
def index_stats():
    """Return statistics about the RAG music index."""
    try:
        from scripts.core.music_index import get_index
        return get_index().get_stats()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/index/rebuild", response_model=JobResponse, status_code=202, tags=["index"])
def rebuild_index(background_tasks: BackgroundTasks = None):
    """
    Rebuild the RAG vector index for all library songs.
    Runs as an async job — poll /jobs/{job_id} for progress.
    """
    _check_job_cap()
    job_id = job_store.create_job(JobType.ANALYZE, {"type": "index_rebuild"})
    job_store.submit_job(job_id, task_rebuild_index)
    return job_store.job_to_response(job_store.get_job(job_id))
