"""
scripts/core/analysis_pipeline.py — Shared "analyze one song" pipeline.

Extracted from scripts/api/task_modules/analysis.py:task_analyze so the same
logic can run two ways:
  1. On demand, via POST /analyze (task_analyze wraps this with job progress).
  2. Automatically right after a download finishes (task_download /
     task_playlist_download call this directly when auto_analyze=True), so
     a song is fully remix-ready — stems split AND analyzed — the moment
     it lands in the library, with no separate manual "Analyze" step.

Writes BOTH cache files so every consumer in the codebase is satisfied:
  - meta.json     — bpm/genre/key cache read by recommend.py / music_index.py
  - analysis.json — phrase boundaries + bpm read by cue_export.py / the
                    /library/{name}/export-cues route (previously NEVER
                    written by anything, so cue export silently fell back
                    to "no analysis" for every song — fixed here as a
                    direct side effect of centralizing this logic).
"""

from typing import Any, Callable, Dict, Optional

from scripts.core.logging_utils import get_logger
from scripts.core.paths import song_dir

_log = get_logger(__name__)

ProgressCB = Optional[Callable[[float, str], None]]


_STEM_NAMES = ("vocals", "drums", "bass", "other")


def _load_song_audio(song: str, sr: int, duration: float):
    """
    Load mono audio for a library song, mono-mixed down to `sr`.

    Prefers full.wav, but falls back to summing whatever Demucs stems are
    present. This matters because prune_on_download (default True) deletes
    full.wav right after stems are produced — so for most songs in the
    library, full.wav simply doesn't exist anymore. Without this fallback,
    every analysis call on an already-pruned song silently fails, which is
    exactly what was happening to /compatibility: it has no way to redo
    analysis, so it fell straight through to the external metadata API
    (which returns all-defaults with no API key configured), producing a
    compatibility score built entirely out of zeros that *looked* real.

    Raises FileNotFoundError if neither full.wav nor any stem exists.
    """
    import librosa
    import numpy as np

    d = song_dir(song)
    full_wav = d / "full.wav"
    if full_wav.exists():
        audio, _ = librosa.load(str(full_wav), sr=sr, mono=True, duration=duration)
        return audio

    mix = None
    for stem in _STEM_NAMES:
        for ext in (".flac", ".wav"):
            p = d / f"{stem}{ext}"
            if p.exists():
                stem_audio, _ = librosa.load(str(p), sr=sr, mono=True, duration=duration)
                mix = stem_audio if mix is None else _add_clipped(mix, stem_audio)
                break

    if mix is None:
        raise FileNotFoundError(f"Song not in library (no full.wav or stems): {song}")
    return mix


def _add_clipped(a, b):
    """Sum two audio arrays of possibly different length, padding the shorter."""
    import numpy as np
    n = max(len(a), len(b))
    out = np.zeros(n, dtype="float32")
    out[: len(a)] += a
    out[: len(b)] += b
    return out


def has_analysis(song_d) -> bool:
    """
    True if a song has been through the REAL run_song_analysis() pipeline —
    not just music_index.py's lightweight upsert_song()/_quick_features()
    fallback, which writes its own partial meta.json (bpm/key/mode/energy/
    danceability/spectral_centroid_hz/vocal_density/chroma_vector) to build
    a search vector for un-indexed songs, but deliberately never computes
    camelot, genre, or analysis.json.

    Checking bpm alone used to be the only test here, but _quick_features()
    also sets bpm — so any song the index quietly upserted during stem
    separation (task_initialize_library calls _index_upsert() right after
    Demucs, long before any genre/key analysis runs) was permanently
    misreported as "fully analyzed" on Library Atlas / processing-status,
    forever missing camelot/genre/duration with no way to re-trigger
    analysis (it never showed up in the "missing" bucket).

    analysis.json is written ONLY by run_song_analysis(), so requiring it
    too is the authoritative signal that real analysis actually ran.
    """
    meta_path = song_d / "meta.json"
    analysis_path = song_d / "analysis.json"
    if not meta_path.exists() or not analysis_path.exists():
        return False
    try:
        import json
        meta = json.loads(meta_path.read_text())
        return "bpm" in meta and meta.get("bpm") is not None
    except Exception:
        return False


def run_song_analysis(
    song: str,
    key_profile: str = "auto",
    progress_cb: ProgressCB = None,
) -> Dict[str, Any]:
    """
    Run genre + structure analysis for a library song and persist the
    results to meta.json and analysis.json. Returns the same result dict
    shape POST /analyze has always returned.

    Works whether or not full.wav still exists — falls back to summing
    Demucs stems (see _load_song_audio) since prune_on_download removes
    full.wav by default right after stems are produced. Raises
    FileNotFoundError only if the song has neither full.wav nor any stems
    (i.e. it isn't really in the library — call this after a download/stem
    job has completed).
    """
    from scripts.core.genre import detect_genre
    from scripts.core.dj_analysis import analyze_structure

    def _progress(frac: float, msg: str) -> None:
        if progress_cb:
            progress_cb(frac, msg)

    _progress(0.1, "Loading audio…")

    sr = 44100
    audio = _load_song_audio(song, sr=sr, duration=120.0)

    _progress(0.4, "Detecting genre…")
    genre = detect_genre(audio, sr)

    _progress(0.7, "Analysing structure…")
    struct = analyze_structure(audio, sr, key_profile=key_profile)

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
        _meta = {k: v for k, v in _meta.items() if v is not None}
        _meta_path.write_text(_json.dumps(_meta))
    except Exception:
        pass

    # ── Persist analysis.json for cue_export.py / export-cues route ───────────
    try:
        import json as _json
        phrase_boundaries = [
            round(s.start_time, 2) for s in struct.sections
        ] if getattr(struct, "sections", None) else []
        analysis_doc = {
            "bpm":                round(struct.bpm, 1),
            "phrase_boundaries":  phrase_boundaries,
            "duration":           round(struct.duration, 1),
            "key":                struct.key_name,
            "camelot":            struct.camelot,
        }
        (song_dir(song) / "analysis.json").write_text(_json.dumps(analysis_doc))
    except Exception as exc:
        _log.warning(f"Failed to write analysis.json for '{song}' (non-critical): {exc}")

    # ── Upsert into RAG music index ────────────────────────────────────────
    try:
        from scripts.core.music_index import get_index
        get_index().upsert_song(song)
    except Exception as exc:
        _log.warning(f"music_index.upsert_song('{song}') failed (non-critical): {exc}")

    result: Dict[str, Any] = {
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

    _progress(1.0, "Analysis complete")
    return result
