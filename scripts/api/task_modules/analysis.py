"""
scripts/api/task_modules/analysis.py — Music analysis and indexing task functions.

task_analyze         — genre + structure analysis
task_rebuild_index   — rebuild RAG music index
task_initialize_library — one-shot pipeline (stem split → compress → index)
"""

from typing import Any, Dict, List, Optional

from scripts.api.jobs import update_job
from scripts.core.audit import log_audit
from scripts.core.logging_utils import get_logger
from scripts.core.paths import LIBRARY_DIR, song_dir

_log = get_logger(__name__)


def _index_upsert(song_name: str) -> None:
    """Non-blocking helper: upsert a song into the RAG music index."""
    try:
        from scripts.core.music_index import get_index
        get_index().upsert_song(song_name)
    except Exception as exc:
        _log.warning(f"music_index.upsert_song('{song_name}') failed (non-critical): {exc}")


def task_analyze(
    job_id: str,
    song: str,
    key_profile: str = "auto",
) -> Dict[str, Any]:
    from scripts.core.analysis_pipeline import run_song_analysis

    log_audit("analyze_start", resource=song, job_id=job_id)

    def _cb(frac: float, msg: str) -> None:
        update_job(job_id, progress=frac, message=msg)

    result = run_song_analysis(song, key_profile=key_profile, progress_cb=_cb)

    log_audit("analyze_complete", resource=song, job_id=job_id,
              metadata={"genre": result.get("genre"), "bpm": result.get("bpm"),
                        "key": result.get("key"), "camelot": result.get("camelot")})
    return result


def task_analyze_missing(job_id: str, key_profile: str = "auto") -> Dict[str, Any]:
    """
    Batch-analyze every library song that's missing BPM/key/energy data
    (i.e. ``has_analysis(song_dir)`` is False — no usable ``meta.json`` yet).

    Best-effort: one song failing doesn't abort the batch. This is the
    backend for the "Analyze all missing" button on Library Atlas — before
    this, filling gaps meant clicking Analyze per-row, one song at a time.
    """
    from scripts.core.analysis_pipeline import has_analysis, run_song_analysis

    targets = [
        d.name for d in sorted(LIBRARY_DIR.iterdir())
        if d.is_dir() and not has_analysis(d)
    ]
    total = len(targets)
    done = 0
    failed: List[Dict[str, str]] = []

    log_audit("analyze_missing_start", resource=f"{total} songs", job_id=job_id)
    update_job(job_id, progress=0.01, message=f"Analyzing {total} song(s) missing data…")

    for i, name in enumerate(targets):
        base = i / max(total, 1)
        step = 1.0 / max(total, 1)

        def _cb(frac: float, msg: str, _base=base, _step=step, _i=i, _name=name) -> None:
            prog = _base + _step * frac
            update_job(job_id, progress=round(min(prog, 0.99), 3),
                       message=f"[{_i + 1}/{total}] {_name[:40]} — {msg}")

        try:
            run_song_analysis(name, key_profile=key_profile, progress_cb=_cb)
            _index_upsert(name)
            done += 1
        except Exception as exc:
            failed.append({"song": name, "error": str(exc)})
            _log.warning(f"analyze_missing: '{name}' failed (non-critical): {exc}")

    summary = f"Done — {done}/{total} analyzed"
    if failed:
        summary += f", {len(failed)} failed"
    update_job(job_id, progress=1.0, message=summary)
    log_audit("analyze_missing_complete", resource=f"{total} songs", job_id=job_id,
              metadata={"done": done, "failed": len(failed)})
    return {"total": total, "done": done, "failed": failed, "success": True}


def task_rebuild_index(job_id: str) -> Dict[str, Any]:
    """
    Rebuild the RAG music index for every song in the library.
    Progress is reported back to the job store in real time.
    """
    from scripts.core.music_index import get_index

    def _cb(frac: float, msg: str) -> None:
        update_job(job_id, progress=round(frac * 0.98, 3), message=msg)

    update_job(job_id, progress=0.01, message="Starting index rebuild…")
    stats = get_index().rebuild(progress_cb=_cb)
    update_job(job_id, progress=1.0, message=(
        f"Done — {stats['indexed']}/{stats['total']} songs indexed "
        f"in {stats['duration_sec']}s"
    ))
    return {"success": True, **stats}


def task_initialize_library(
    job_id: str,
    enhance: bool = True,
    model: str = "htdemucs",
    delete_wav: bool = True,
    run_compress: bool = True,
    run_index: bool = True,
) -> Dict[str, Any]:
    """
    One-shot pipeline:
      1. Batch Demucs stem separation on all songs that lack stems
      2. Batch FLAC compression of all new WAV stems
      3. Rebuild the RAG music index

    Progress is mapped across three phases:
      0–65 %  →  stem separation
      65–85 % →  FLAC compression
      85–100% →  index rebuild
    """
    from scripts.core.stems import separate_song_stems
    from scripts.core.music_index import get_index

    STEMS = ("vocals", "drums", "bass", "other")

    # ── Count library before starting ────────────────────────────────────────
    try:
        _all_songs = [d.name for d in LIBRARY_DIR.iterdir() if d.is_dir()]
    except Exception:
        _all_songs = []
    log_audit("initialize_library_start", resource=f"{len(_all_songs)} songs", job_id=job_id,
              metadata={"enhance": enhance, "model": model, "run_compress": run_compress,
                        "run_index": run_index})

    # ── Phase 1: stem separation ──────────────────────────────────────────────
    songs_needing_stems = [
        d.name for d in sorted(LIBRARY_DIR.iterdir())
        if d.is_dir() and not all((d / f"{s}.wav").exists() or (d / f"{s}.flac").exists()
                                  for s in STEMS)
    ]
    total_split   = len(songs_needing_stems)
    split_done    = 0
    split_skipped = 0
    split_failed  = []

    update_job(job_id, progress=0.01,
               message=f"Phase 1/3 · Stem separation — {total_split} songs to process")

    for i, song_name in enumerate(songs_needing_stems):
        base_prog = 0.01 + 0.63 * (i / max(total_split, 1))
        update_job(job_id, progress=round(base_prog, 3),
                   message=f"[{i+1}/{total_split}] Demucs · {song_name[:42]}")
        try:
            separate_song_stems(song_name, enhance=enhance, model=model)
            _index_upsert(song_name)
            split_done += 1
        except Exception as exc:
            split_failed.append({"song": song_name, "error": str(exc)})

    update_job(job_id, progress=0.64,
               message=f"Stem separation done — {split_done} split, {len(split_failed)} failed")

    # ── Phase 2: FLAC compression ─────────────────────────────────────────────
    compress_converted = 0
    compress_skipped   = 0
    compress_failed    = []

    if run_compress:
        songs_with_stems = [
            d.name for d in sorted(LIBRARY_DIR.iterdir())
            if d.is_dir() and any((d / f"{s}.wav").exists() for s in STEMS)
        ]
        total_compress = len(songs_with_stems)
        update_job(job_id, progress=0.65,
                   message=f"Phase 2/3 · FLAC compression — {total_compress} songs")

        for i, song_name in enumerate(songs_with_stems):
            prog = 0.65 + 0.19 * (i / max(total_compress, 1))
            update_job(job_id, progress=round(prog, 3),
                       message=f"[{i+1}/{total_compress}] Compressing · {song_name[:42]}")
            try:
                import soundfile as sf
                d = song_dir(song_name)
                did_compress = False
                for stem in STEMS:
                    wav_p  = d / f"{stem}.wav"
                    flac_p = d / f"{stem}.flac"
                    if not wav_p.exists() or flac_p.exists():
                        continue
                    data, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
                    sf.write(str(flac_p), data, sr, format="flac", subtype="PCM_24")
                    if delete_wav and flac_p.exists():
                        wav_p.unlink()
                    did_compress = True
                if did_compress:
                    compress_converted += 1
                else:
                    compress_skipped += 1
            except Exception as exc:
                compress_failed.append({"song": song_name, "error": str(exc)})

        update_job(job_id, progress=0.84,
                   message=f"Compression done — {compress_converted} compressed, "
                           f"{compress_skipped} skipped")

    # ── Phase 3: index rebuild ────────────────────────────────────────────────
    index_stats: Dict[str, Any] = {}
    if run_index:
        update_job(job_id, progress=0.85, message="Phase 3/3 · Rebuilding RAG index…")

        def _idx_cb(frac: float, msg: str) -> None:
            prog = 0.85 + 0.14 * frac
            update_job(job_id, progress=round(prog, 3), message=f"Index · {msg}")

        index_stats = get_index().rebuild(progress_cb=_idx_cb)

    update_job(job_id, progress=1.0,
               message=f"✅ Done — {split_done} stems split · {compress_converted} compressed · "
                       f"{index_stats.get('indexed', '?')} indexed")
    log_audit("initialize_library_complete", resource=f"{len(_all_songs)} songs", job_id=job_id,
              metadata={"split_done": split_done, "compress_converted": compress_converted,
                        "total_indexed": index_stats.get("indexed", 0)})
    return {
        "split_done":          split_done,
        "split_skipped":       split_skipped,
        "split_failed":        len(split_failed),
        "compress_converted":  compress_converted,
        "compress_skipped":    compress_skipped,
        "compress_failed":     len(compress_failed),
        "total_indexed":       index_stats.get("indexed", 0),
        "success":             True,
    }
