"""
scripts/api/task_modules/download.py — Track download task functions.

task_download          — download a single track with automatic Demucs
task_playlist_download — download all tracks from a playlist
"""

from pathlib import Path
from typing import Any, Dict, Optional

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.logging_utils import get_logger

_log = get_logger(__name__)


def _index_upsert(song_name: str) -> None:
    """Non-blocking helper: upsert a song into the RAG music index."""
    try:
        from scripts.core.music_index import get_index
        get_index().upsert_song(song_name)
    except Exception as exc:
        _log.warning(f"music_index.upsert_song('{song_name}') failed (non-critical): {exc}")


def task_download(
    job_id: str,
    query: str,
    name: Optional[str] = None,
    separate: bool = True,   # always True now — Demucs is automatic
    auto_analyze: bool = True,
) -> Dict[str, Any]:
    from scripts.download import download_track, TrackSpec

    log_audit("download_start", resource=query, job_id=job_id, metadata={"separate": separate})
    update_job(job_id, progress=0.01, message="Starting download…")

    def _progress(frac: float, msg: str) -> None:
        # download_track's own pipeline (download → stems → auto-analyze →
        # prune → register) covers the full 0–1 range; auto_analyze runs
        # *inside* it, before the raw WAV is pruned, so it actually has a
        # full.wav to read. Don't try to re-run analysis after the fact here
        # — by then prune_on_download may have already deleted full.wav.
        update_job(job_id, progress=round(0.01 + 0.95 * frac, 3), message=msg)

    # Force separate=True — every download gets Demucs automatically
    spec = TrackSpec(query=query, name=name, separate=True, auto_analyze=auto_analyze)
    result = download_track(spec, progress_cb=_progress)

    if not result.success:
        log_audit("download_failed", resource=query, job_id=job_id, metadata={"error": result.error})
        raise RuntimeError(result.error or "Download failed")

    update_job(job_id, progress=0.97, message="Indexing song…")

    # ── Upsert into music index (non-blocking) ─────────────────────────────
    _index_upsert(result.name)

    out: Dict[str, Any] = {
        "name":     result.name,
        "wav":      str(result.wav) if result.wav else None,
        "stems":    {k: str(v) for k, v in result.stems.items()},
        "indexed":  True,
        "success":  True,
        "analyzed": result.analyzed,
    }
    if result.license_warning:
        out["license_warning"] = result.license_warning
    if result.evicted:
        # Storage cap was exceeded — these OTHER songs got their full.wav
        # (or whole directory) auto-removed to make room. Surface it so
        # it's never a silent surprise.
        out["evicted"] = result.evicted

    update_job(job_id, progress=1.0, message="Finalising…")

    log_audit("download_complete", resource=result.name, job_id=job_id,
              metadata={"query": query, "stems": list(result.stems.keys()),
                        "analyzed": result.analyzed})
    return out


def task_playlist_download(
    job_id: str,
    url: str,
    separate: bool,
    limit: Optional[int],
    auto_analyze: bool = True,
) -> Dict[str, Any]:
    """
    Download all tracks from a playlist URL.
    Reports per-track progress back to the job store.
    """
    from scripts.download import _sanitize, TrackSpec, download_track, DownloadResult

    update_job(job_id, progress=0.02, message="Fetching playlist info…")

    # ── Extract playlist entries via yt-dlp (no download yet) ────────────
    try:
        import yt_dlp  # type: ignore
        info_opts = {
            "quiet": True,
            "extract_flat": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(info_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        raise RuntimeError(f"Could not fetch playlist: {e}")

    entries = info.get("entries") or []
    if not entries:
        raise RuntimeError("Playlist appears to be empty or is not accessible.")

    playlist_title = info.get("title") or "Playlist"

    # Apply optional track limit
    if limit:
        entries = entries[:limit]

    total = len(entries)
    update_job(job_id, progress=0.05, message=f"Found {total} tracks in '{playlist_title}'")

    results = []
    failed  = []

    for i, entry in enumerate(entries):
        vid       = entry.get("id") or entry.get("url", "")
        title     = _sanitize(entry.get("title") or f"track_{i+1:03d}")
        track_url = (
            entry.get("webpage_url")
            or entry.get("url")
            or f"https://www.youtube.com/watch?v={vid}"
        )

        base = 0.05 + 0.93 * (i / total)
        span = 0.93 / total
        update_job(
            job_id,
            progress=round(base, 3),
            message=f"[{i+1}/{total}] Downloading: {title[:40]}…",
        )

        def _track_progress(frac: float, msg: str) -> None:
            update_job(
                job_id,
                progress=round(base + span * frac, 3),
                message=f"[{i+1}/{total}] {title[:40]}: {msg}",
            )

        # auto_analyze runs inside download_track(), before the raw WAV gets
        # pruned — see task_download's comment for why ordering matters here.
        spec   = TrackSpec(query=track_url, name=title, separate=separate, auto_analyze=auto_analyze)
        result = download_track(spec, progress_cb=_track_progress)

        if result.success:
            results.append({
                "name":     result.name,
                "wav":      str(result.wav) if result.wav else None,
                "stems":    {k: str(v) for k, v in result.stems.items()},
                "success":  True,
                "analyzed": result.analyzed,
            })
        else:
            failed.append({"name": title, "error": result.error})

    update_job(job_id, progress=0.99, message=f"Done — {len(results)}/{total} tracks saved")

    return {
        "playlist_title": playlist_title,
        "total":          total,
        "downloaded":     len(results),
        "failed":         len(failed),
        "tracks":         results,
        "errors":         failed,
        "success":        True,
    }
