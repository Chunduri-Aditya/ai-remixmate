"""
scripts/api/task_modules/analysis.py — Music analysis and indexing task functions.

task_analyze         — genre + structure analysis
task_rebuild_index   — rebuild RAG music index
task_initialize_library — one-shot pipeline (stem split → compress → index)
"""

from typing import Any, Dict, List, Optional

import librosa

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
) -> Dict[str, Any]:
    from scripts.core.genre import detect_genre
    from scripts.core.dj_engine import _analyze_impl

    log_audit("analyze_start", resource=song, job_id=job_id)
    update_job(job_id, progress=0.1, message="Loading audio…")

    wav = song_dir(song) / "full.wav"
    if not wav.exists():
        raise FileNotFoundError(f"Song not in library: {song}")

    sr = 44100
    audio, _ = librosa.load(str(wav), sr=sr, mono=True, duration=120.0)

    update_job(job_id, progress=0.4, message="Detecting genre…")
    genre = detect_genre(audio, sr)

    update_job(job_id, progress=0.7, message="Analysing structure…")
    struct = _analyze_impl(audio, sr)

    # ── Persist BPM + genre to meta.json cache for the recommendation engine ──
    try:
        from scripts.core.recommend import write_meta_cache
        write_meta_cache(song_dir(song), bpm=struct.bpm, genre=genre.genre)
    except Exception:
        pass  # Non-critical — recommendation engine will fall back to quick BPM

    # ── Persist full music intelligence fields to meta.json for RAG index ──
    try:
        import json as _json
        _meta_path = song_dir(song) / "meta.json"
        _meta = {}
        if _meta_path.exists():
            _meta = _json.loads(_meta_path.read_text())
        _meta.update({
            "bpm":                   round(struct.bpm, 1),
            "genre":                 genre.genre,
            "key":                   struct.key_name,
            "mode":                  struct.mode,
            "camelot":               struct.camelot,
            "energy_mean":           round(struct.energy_mean, 4)     if hasattr(struct, "energy_mean") else None,
            "energy_std":            round(struct.energy_std,  4)     if hasattr(struct, "energy_std")  else None,
            "danceability":          round(struct.danceability, 4)    if hasattr(struct, "danceability") else None,
            "beat_strength":         round(struct.beat_strength, 4)   if hasattr(struct, "beat_strength") else None,
            "tempo_stability":       round(struct.tempo_stability, 4) if hasattr(struct, "tempo_stability") else None,
            "vocal_density":         round(struct.vocal_density, 4)   if hasattr(struct, "vocal_density") else None,
            "spectral_centroid_hz":  round(struct.spectral_centroid_hz, 1) if hasattr(struct, "spectral_centroid_hz") else None,
            "chroma_vector":         struct.chord_sequence if hasattr(struct, "chroma_vector") else None,
        })
        # Remove None values
        _meta = {k: v for k, v in _meta.items() if v is not None}
        _meta_path.write_text(_json.dumps(_meta))
    except Exception:
        pass

    # ── Upsert into RAG music index ────────────────────────────────────────
    _index_upsert(song)

    result: dict = {
        "song":       song,
        "genre":      genre.genre,
        "confidence": round(genre.confidence, 3),
        "runner_up":  genre.runner_up,
        "bpm":        round(struct.bpm, 1),
        "total_bars": struct.total_bars,
        "duration":   round(struct.duration, 1),
        "sections":   [
            {
                "type":       s.type,
                "start_bar":  s.start_bar,
                "end_bar":    s.end_bar,
                "start_time": round(s.start_time, 2),
                "end_time":   round(s.end_time, 2),
            }
            for s in struct.sections
        ],
    }

    # Music intelligence fields (populated by _analyze_impl enrichment)
    if struct.key_name:
        result["key"]              = struct.key_name
        result["mode"]             = struct.mode
        result["camelot"]          = struct.camelot
        result["key_confidence"]   = round(struct.key_confidence, 3)
        result["danceability"]     = round(struct.danceability, 3)
        result["vocal_density"]    = round(struct.vocal_density, 3)
        result["spectral_centroid_hz"] = round(struct.spectral_centroid_hz, 1)
        result["chord_sequence"]   = struct.chord_sequence
        if struct.drop_position is not None:
            result["drop_position_sec"] = round(struct.drop_position, 1)

    log_audit("analyze_complete", resource=song, job_id=job_id,
              metadata={"genre": genre.genre, "bpm": round(struct.bpm, 1),
                        "key": struct.key_name, "camelot": struct.camelot})
    return result


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
