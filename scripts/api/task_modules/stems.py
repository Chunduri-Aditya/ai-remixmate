"""
scripts/api/task_modules/stems.py — Stem separation and compression task functions.

task_stem_split             — stem-split a single song
task_batch_stem_split       — stem-split multiple songs
task_compress_stems         — compress stems for a single song
task_batch_compress_stems   — compress stems for all library songs
"""

from typing import Any, Dict, List, Optional

from scripts.api.jobs import update_job
from scripts.core.paths import LIBRARY_DIR, song_dir

_STEMS = ("vocals", "drums", "bass", "other")


def task_stem_split(
    job_id: str,
    song: str,
    enhance: bool = True,
    model: str = "htdemucs",
) -> Dict[str, Any]:
    """
    Enhance then Demucs-split a single library song.
    All logic lives in scripts/core/stems.py — this task just wires
    progress callbacks to the job store.
    """
    from scripts.core.stems import separate_song_stems

    def _progress(prog: float, msg: str) -> None:
        update_job(job_id, progress=round(prog, 3), message=msg)

    result = separate_song_stems(
        song_name   = song,
        enhance     = enhance,
        model       = model,
        progress_cb = _progress,
    )

    if not result.success:
        raise RuntimeError(result.error or "Stem separation failed")

    return {
        "song":         result.song,
        "stems":        result.stem_info,
        "n_stems":      len(result.stems),
        "enhance_info": result.enhance_info,
        "model":        result.model,
        "success":      True,
    }


def task_batch_stem_split(
    job_id: str,
    songs: list,
    enhance: bool = True,
    model: str = "htdemucs",
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Run Demucs on a list of songs one by one.
    Reports per-song progress; a failed song doesn't abort the batch.
    """
    from scripts.core.stems import separate_song_stems

    total   = len(songs)
    results = []
    failed  = []

    update_job(job_id, progress=0.01, message=f"Starting batch split: {total} songs…")

    for i, song in enumerate(songs):
        base_prog = i / total
        next_prog = (i + 1) / total

        # ── Skip songs that already have all 4 stems (WAV or FLAC) ──────
        _sd = song_dir(song)
        if skip_existing and (
            (_sd / "vocals.flac").exists() or (_sd / "vocals.wav").exists()
        ):
            update_job(job_id,
                       progress=round(next_prog, 3),
                       message=f"[{i+1}/{total}] Skipped (stems exist): {song[:40]}")
            results.append({"song": song, "skipped": True,
                            "stems": ["vocals", "drums", "bass", "other"]})
            continue

        # ── Progress callback — scales individual progress into batch slot ─
        def _progress(prog: float, msg: str,
                      _base=base_prog, _next=next_prog, _i=i+1) -> None:
            batch_prog = _base + (_next - _base) * prog
            update_job(job_id,
                       progress=round(batch_prog, 3),
                       message=f"[{_i}/{total}] {msg}")

        update_job(job_id,
                   progress=round(base_prog, 3),
                   message=f"[{i+1}/{total}] Processing: {song[:40]}…")

        result = separate_song_stems(
            song_name   = song,
            enhance     = enhance,
            model       = model,
            progress_cb = _progress,
        )

        if result.success:
            results.append({
                "song":    song,
                "stems":   list(result.stems.keys()),
                "success": True,
            })
            update_job(job_id,
                       progress=round(next_prog, 3),
                       message=f"[{i+1}/{total}] Done: {song[:40]} ✓")
        else:
            failed.append({"song": song, "error": result.error})
            update_job(job_id,
                       progress=round(next_prog, 3),
                       message=f"[{i+1}/{total}] Failed: {song[:40]} — {(result.error or '')[:60]}")

    update_job(job_id, progress=0.99,
               message=f"Batch done — {len(results)}/{total} processed, {len(failed)} failed")

    return {
        "total":     total,
        "processed": len(results),
        "failed":    len(failed),
        "results":   results,
        "errors":    failed,
        "success":   True,
    }


def task_compress_stems(
    job_id: str,
    song: str,
    delete_wav: bool = True,
) -> Dict[str, Any]:
    """
    Convert WAV stems (vocals / drums / bass / other) to lossless FLAC.

    FLAC gives ~50 % space savings vs WAV with zero quality loss.
    Fast decode means retrieval speed is virtually unchanged.

    Parameters
    ----------
    delete_wav  : bool
        If True, delete the source WAV after a successful FLAC encode.
    """
    import soundfile as sf

    d = song_dir(song)
    converted: List[str] = []
    skipped:   List[str] = []
    failed:    List[str] = []

    total = len(_STEMS)
    for i, stem in enumerate(_STEMS):
        update_job(job_id,
                   progress=round(i / total, 3),
                   message=f"Compressing {stem} stem…")

        wav_path  = d / f"{stem}.wav"
        flac_path = d / f"{stem}.flac"

        if flac_path.exists():
            skipped.append(stem)
            continue
        if not wav_path.exists():
            skipped.append(stem)
            continue

        try:
            data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
            sf.write(str(flac_path), data, sr, format="flac",
                     subtype="PCM_24")

            if delete_wav and flac_path.exists():
                wav_path.unlink()

            converted.append(stem)
        except Exception as exc:
            failed.append(f"{stem}: {exc}")

    update_job(job_id, progress=0.99,
               message=f"Compressed {len(converted)} stems; "
                       f"skipped {len(skipped)}, failed {len(failed)}")
    return {
        "song":       song,
        "converted":  converted,
        "skipped":    skipped,
        "failed":     failed,
        "success":    len(failed) == 0,
    }


def task_batch_compress_stems(
    job_id: str,
    songs: Optional[List[str]] = None,
    delete_wav: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Any]:
    """
    Convert all stem WAVs in the library to lossless FLAC.

    Runs ``task_compress_stems`` per song.  Songs without any stems are
    skipped silently.  A failed song does NOT abort the batch.

    Parameters
    ----------
    songs       : list of song names, or None → all library songs
    delete_wav  : delete WAV after successful FLAC encode
    skip_existing : skip songs where all FLAC files already exist
    """
    if songs is None:
        songs = [
            d.name for d in sorted(LIBRARY_DIR.iterdir())
            if d.is_dir() and any((d / f"{s}.wav").exists() for s in _STEMS)
        ]

    total     = len(songs)
    converted = 0
    skipped   = 0
    failed    = []

    update_job(job_id, progress=0.01,
               message=f"Batch compress starting — {total} songs with stems")

    for i, song_name in enumerate(songs):
        prog = 0.01 + 0.97 * (i / max(total, 1))
        update_job(job_id, progress=round(prog, 3),
                   message=f"[{i+1}/{total}] Compressing: {song_name[:40]}")

        d = song_dir(song_name)

        if skip_existing and all((d / f"{s}.flac").exists() for s in _STEMS):
            skipped += 1
            continue

        try:
            import soundfile as sf
            song_converted = 0
            for stem in _STEMS:
                wav_p  = d / f"{stem}.wav"
                flac_p = d / f"{stem}.flac"
                if not wav_p.exists() or (skip_existing and flac_p.exists()):
                    continue
                data, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
                sf.write(str(flac_p), data, sr, format="flac", subtype="PCM_24")
                if delete_wav and flac_p.exists():
                    wav_p.unlink()
                song_converted += 1
            converted += 1
        except Exception as exc:
            failed.append({"song": song_name, "error": str(exc)})

    update_job(job_id, progress=1.0,
               message=f"Done — {converted} songs compressed, "
                       f"{skipped} skipped, {len(failed)} failed")
    return {
        "total":     total,
        "converted": converted,
        "skipped":   skipped,
        "failed":    len(failed),
        "errors":    failed[:10],
        "success":   True,
    }
