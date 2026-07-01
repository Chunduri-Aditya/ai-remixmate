"""
scripts/api/routers/analysis.py — Music analysis and recommendation endpoints.

POST /compatibility                    — instant Camelot + BPM check
POST /analyze                          — genre + structure analysis (queued job)
GET  /recommend/{name}                 — top compatible songs for a track
GET  /library/similar/{name}           — RAG vector similarity search
GET  /library/crate-search             — CLAP 512-D semantic similarity search
GET  /index/stats                      — RAG index statistics
POST /index/rebuild                    — rebuild RAG index (queued job)
POST /library/build-clap-index         — build / update CLAP embedding index (queued job)
"""

from typing import Any, Dict, List, Optional

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
from scripts.core.logging_utils import get_logger
from scripts.core.paths import LIBRARY_DIR

_log = get_logger(__name__)

router = APIRouter()


@router.post("/compatibility", response_model=CompatibilityResult, tags=["analysis"])
def check_compatibility(req: CompatibilityRequest):
    """
    Instant Camelot + BPM compatibility check.

    Order of preference per song:
      1. Cached analysis (meta.json, written by POST /analyze or an
         auto-analyzed download) — fast, no audio decode.
      2. Live analysis on whatever audio exists in the library — full.wav
         if present, or the Demucs stems summed back together if it was
         already pruned (prune_on_download deletes full.wav by default,
         right after stems are produced, which is the common case).
      3. External metadata API (getsongbpm/last.fm), only for songs that
         aren't in the local library at all.

    Previously this only checked full.wav, which prune_on_download deletes
    almost immediately — so virtually every library song fell through to
    the external API. With no API key configured by default, that returns
    an all-defaults TrackMetadata (bpm=0.0, camelot=None), and the score
    computed from two of those looks like a real result (e.g. "57/100")
    while actually being a comparison of two empty objects.
    """
    from scripts.core.track_metadata import TrackMetadata, MetadataClient
    from scripts.core.analysis_pipeline import has_analysis, run_song_analysis

    client = MetadataClient()

    def _meta_for(name: str, artist: str) -> tuple[TrackMetadata, Optional[str]]:
        d = LIBRARY_DIR / name
        if d.exists():
            # 1. Cached analysis — fast path.
            if has_analysis(d):
                try:
                    import json as _json
                    cached = _json.loads((d / "meta.json").read_text())
                    meta = TrackMetadata(
                        title=name, artist=artist,
                        bpm=float(cached.get("bpm") or 0.0),
                        key=cached.get("key") or "",
                        mode=cached.get("mode") or "",
                        camelot=cached.get("camelot"),
                        genres=[cached["genre"]] if cached.get("genre") else [],
                        # energy_mean/vocal_density/spectral_centroid_hz are
                        # written to meta.json by run_song_analysis() but were
                        # never read back here — energy defaulted to 0.5 for
                        # every library song, silently making energy_score
                        # compare 0.5-vs-0.5 (always 1.0) for any pair, and
                        # genre_proximity/timbral_similarity/vocal_clash_penalty
                        # had no data to work with at all. Fixed alongside
                        # wiring those three into compatibility_score()
                        # (REMIX_QUALITY_INSIGHTS.md finding #1).
                        energy=float(cached["energy_mean"]) if cached.get("energy_mean") is not None else 0.5,
                        vocal_density=float(cached.get("vocal_density") or 0.0),
                        spectral_centroid_hz=float(cached.get("spectral_centroid_hz") or 0.0),
                    )
                    return meta, cached.get("genre")
                except Exception as exc:
                    _log.warning(f"compatibility: cached meta.json unreadable for '{name}': {exc}")

            # 2. No usable cache — analyze now (stem-aware fallback covers
            #    the post-prune case) and cache the result for next time.
            try:
                result = run_song_analysis(name)
                # run_song_analysis()'s return dict doesn't carry
                # energy_mean/vocal_density/spectral_centroid_hz (it writes
                # them straight to meta.json instead) — re-read the cache it
                # just wrote rather than duplicating that computation here.
                cached_extra: Dict[str, Any] = {}
                try:
                    import json as _json
                    cached_extra = _json.loads((d / "meta.json").read_text())
                except Exception:
                    pass
                meta = TrackMetadata(
                    title=name, artist=artist,
                    bpm=float(result.get("bpm") or 0.0),
                    key=result.get("key") or "",
                    mode=result.get("mode") or "",
                    camelot=result.get("camelot"),
                    genres=[result["genre"]] if result.get("genre") else [],
                    energy=float(cached_extra["energy_mean"]) if cached_extra.get("energy_mean") is not None else 0.5,
                    vocal_density=float(cached_extra.get("vocal_density") or 0.0),
                    spectral_centroid_hz=float(cached_extra.get("spectral_centroid_hz") or 0.0),
                )
                return meta, result.get("genre")
            except Exception as exc:
                _log.warning(f"compatibility: live analysis failed for '{name}' (non-critical): {exc}")

        # 3. Not in the library (or analysis failed) — external API.
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
        genre_proximity=round(score.get("genre_proximity", 0.5), 3),
        timbral_similarity=round(score.get("timbral_similarity", 0.5), 3),
        vocal_clash_penalty=round(score.get("vocal_clash_penalty", 0.0), 3),
        bpm_a=round(meta_a.bpm, 1),
        bpm_b=round(meta_b.bpm, 1),
        camelot_a=meta_a.camelot,
        camelot_b=meta_b.camelot,
        genre_a=genre_a,
        genre_b=genre_b,
    )


@router.post("/analyze", response_model=JobResponse, status_code=202, tags=["analysis"])
def analyze_song(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks = None,
    key_profile: str = Query(
        "auto",
        description=(
            "Tonal profile for key detection: "
            "'ks' (Krumhansl-Schmuckler), "
            "'edma' (EDM major), "
            "'edmm' (EDM major+minor), "
            "'auto' (spectral-centroid-based selection)"
        ),
    ),
):
    _check_job_cap()
    _require_song(req.song)
    job_id = job_store.create_job(JobType.ANALYZE, {"song": req.song, "key_profile": key_profile})
    job_store.submit_job(job_id, task_analyze, song=req.song, key_profile=key_profile)
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


# ---------------------------------------------------------------------------
# Stage 3A — CLAP Crate Digger (semantic similarity search)
# ---------------------------------------------------------------------------

@router.get("/library/crate-search", tags=["analysis"])
def crate_search(
    q: Optional[str] = Query(None, description="Text query ('dark minimal techno')"),
    name: Optional[str] = Query(None, description="Library song name to use as audio query"),
    k: int = Query(10, ge=1, le=50, description="Number of results"),
    camelot: Optional[str] = Query(None, description="Camelot key filter, e.g. '10A'"),
    bpm_min: Optional[float] = Query(None, description="Minimum BPM filter"),
    bpm_max: Optional[float] = Query(None, description="Maximum BPM filter"),
):
    """
    Semantic similarity search using CLAP 512-D embeddings.

    At least one of ``q`` (text) or ``name`` (song) must be provided.
    When both are supplied, their embeddings are averaged before search.

    Requires ``laion_clap`` (``pip install laion_clap``).  Falls back to the
    35-D music_index.py search when CLAP is not installed; text queries are
    degraded to a name-substring keyword match in that case.

    Returns the top-k results sorted by similarity descending.
    """
    if not q and not name:
        raise HTTPException(status_code=400, detail="Provide at least one of: q (text) or name (song)")

    bpm_range = None
    if bpm_min is not None or bpm_max is not None:
        bpm_range = (float(bpm_min or 0.0), float(bpm_max or 9999.0))

    try:
        from scripts.core.crate_digger import get_digger
        digger = get_digger()
        if digger.is_empty():
            raise HTTPException(
                status_code=404,
                detail="CLAP index is empty — run POST /library/build-clap-index first",
            )
        results = digger.find_similar(
            query_name=name,
            query_text=q,
            k=k,
            camelot_filter=camelot,
            bpm_range=bpm_range,
        )
        return {
            "query_text": q,
            "query_name": name,
            "results": [
                {
                    "name": r.name,
                    "score": round(r.score, 4),
                    "bpm": r.bpm,
                    "key": r.key,
                    "camelot": r.camelot,
                    "energy": r.energy,
                    "backend": r.backend,
                }
                for r in results
            ],
            "backend": results[0].backend if results else "none",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/library/build-clap-index", response_model=JobResponse, status_code=202, tags=["analysis"])
def build_clap_index(
    force: bool = Query(False, description="Re-embed all songs even if already indexed"),
):
    """
    Build (or update) the CLAP 512-D embedding index for the whole library.

    Uses LAION-CLAP (``pip install laion_clap``) when available; falls back
    to the 35-D music_index.py embeddings.

    Already-indexed songs are skipped unless ``force=True``.

    Returns a job_id for SSE tracking.
    """
    _check_job_cap()

    def _task_build_clap(job_id: str) -> None:
        try:
            from scripts.core.crate_digger import get_digger
            job_store.update_job(job_id, status="running", progress=0.05,
                                 message="Initialising CLAP index…")

            def _progress(frac: float):
                job_store.update_job(
                    job_id, progress=max(0.05, min(0.95, frac)),
                    message=f"Embedding songs… {int(frac * 100)}%",
                )

            digger = get_digger()
            n = digger.index_library(progress_cb=_progress, force=force)
            stats = digger.get_stats()
            job_store.update_job(
                job_id, status="done", progress=1.0,
                message=f"CLAP index built: {stats['n_songs']} songs",
                result=stats,
            )
        except Exception as exc:
            job_store.update_job(job_id, status="failed",
                                 message=f"CLAP index build failed: {exc}")

    job_id = job_store.create_job(JobType.ANALYZE, {"type": "build_clap_index", "force": force})
    job_store.submit_job(job_id, _task_build_clap)
    return job_store.job_to_response(job_store.get_job(job_id))
