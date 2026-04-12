"""
scripts/core/stems.py — Canonical stem-separation module for AI RemixMate.

This is the single authoritative function for splitting any library song
into Demucs stems. All callers (tasks, CLI, download pipeline) import from
here so the logic is never duplicated.

Key behaviour
─────────────
• Resolves the correct Python executable (venv-first, so Demucs is always found)
• Optionally runs the audio-enhancement chain before separation
• Handles every Demucs output-directory layout variation
• Normalises each stem to its recommended LUFS target after splitting
• Returns a structured StemResult dataclass

Usage
─────
    from scripts.core.stems import separate_song_stems, StemResult

    result = separate_song_stems("Anyma - Abyss", enhance=True)
    if result.success:
        print(result.stems)          # {"vocals": Path(...), "drums": ..., ...}
        print(result.enhance_info)
    else:
        print(result.error)

CLI
───
    python -m scripts.core.stems "Song Name"
    python -m scripts.core.stems "Song Name" --model htdemucs_ft --no-enhance
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Locate the correct Python / Demucs executable
# ---------------------------------------------------------------------------

def _venv_python() -> str:
    """
    Return the path to the virtual-environment Python that has Demucs installed.

    Search order:
      1. remix-env/bin/python3   relative to the project root (most reliable)
      2. The python3 sibling of the demucs script on PATH
      3. sys.executable  (works when the venv is activated in the calling shell)
    """
    # Project root = two levels up from this file (scripts/core/stems.py)
    project_root = Path(__file__).resolve().parents[2]

    candidates: List[Path] = [
        project_root / "remix-env" / "bin" / "python3",
        project_root / "remix-env" / "bin" / "python",
    ]

    # Also look for a demucs script and grab its shebang Python
    demucs_bin = shutil.which("demucs")
    if demucs_bin:
        try:
            first_line = Path(demucs_bin).read_text().splitlines()[0]
            if first_line.startswith("#!"):
                cand = Path(first_line[2:].strip())
                if cand.exists():
                    candidates.insert(0, cand)
        except Exception:
            pass

    for cand in candidates:
        # Use lstat so broken symlinks still count — they point to a real
        # Python on the user's machine even if the VM can't follow the link.
        try:
            cand.lstat()          # raises if path doesn't exist at all
            return str(cand)
        except (OSError, FileNotFoundError):
            continue

    # Last resort — sys.executable (works if the venv is already activated)
    return sys.executable


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class StemResult:
    song:         str
    success:      bool
    stems:        Dict[str, Path] = field(default_factory=dict)   # name → Path
    stem_info:    Dict[str, dict] = field(default_factory=dict)   # name → {rms_db, path}
    enhance_info: dict            = field(default_factory=dict)
    model:        str             = "htdemucs"
    error:        Optional[str]   = None


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """
    Detect the best available device for Demucs via the centralized GPU module.

    Returns 'mps' on Apple Silicon, 'cuda' if NVIDIA GPU, or 'cpu' as fallback.
    """
    try:
        from scripts.core.gpu import get_device
        return get_device()
    except ImportError:
        import platform
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            return "mps"
        return "cpu"


def separate_song_stems(
    song_name: str,
    enhance: bool      = True,
    model: str         = "htdemucs",
    device: str        = "auto",
    progress_cb=None,                # optional callable(float, str) for progress updates
) -> StemResult:
    """
    Enhance and split a single library song into vocal / drum / bass / other stems.

    Parameters
    ----------
    song_name   : exact folder name inside library/
    enhance     : run audio-enhancement chain before Demucs (recommended)
    model       : Demucs model name  (htdemucs | htdemucs_ft | mdx_extra)
    device      : 'auto' (detect GPU), 'mps', 'cuda', or 'cpu'
    progress_cb : optional callable(progress: float, message: str)

    Returns
    -------
    StemResult with .success, .stems, .enhance_info, .error
    """
    import numpy as np
    import soundfile as sf

    try:
        import librosa
    except ImportError as exc:
        return StemResult(song=song_name, success=False,
                          error=f"librosa not available: {exc}")

    from scripts.core.paths import song_dir as _song_dir

    def _prog(p: float, msg: str) -> None:
        log.info("[stems] %.0f%%  %s", p * 100, msg)
        if progress_cb:
            try:
                progress_cb(p, msg)
            except Exception:
                pass

    dest_dir = _song_dir(song_name)
    wav_path = dest_dir / "full.wav"

    if not wav_path.exists():
        return StemResult(song=song_name, success=False,
                          error=f"full.wav not found in library/{song_name}/")

    # ── 1. Load audio ────────────────────────────────────────────────────
    _prog(0.03, f"Loading {song_name[:40]}…")
    sr = 44100
    try:
        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    except Exception as exc:
        return StemResult(song=song_name, success=False,
                          error=f"Could not load audio: {exc}")

    # ── 2. Enhancement chain ──────────────────────────────────────────────
    demucs_input = wav_path
    enhance_info: dict = {}

    if enhance:
        _prog(0.08, "Enhancing audio (HP · compress · air EQ · LUFS −14)…")
        try:
            from scripts.core.audio_enhance import enhance_audio, EnhanceOptions
            opts = EnhanceOptions(
                hp_filter      = True,
                lp_filter      = True,
                noise_gate     = True,
                compression    = True,
                air_eq         = True,
                lufs_target    = -14.0,
                true_peak_ceil = -1.0,
            )
            enh_audio, enh_report = enhance_audio(audio, sr, opts)
            enh_path = dest_dir / "full_enhanced.wav"
            sf.write(str(enh_path), enh_audio, sr, subtype="PCM_24")
            demucs_input = enh_path
            enhance_info = {
                "lufs_before":  round(enh_report.lufs_before, 1),
                "lufs_after":   round(enh_report.lufs_after, 1),
                "gain_db":      round(enh_report.gain_applied_db, 1),
                "stages":       enh_report.stages_applied,
                "clipped":      enh_report.clipped,
            }
            _prog(0.18, f"Enhanced: {enhance_info['lufs_before']} → {enhance_info['lufs_after']} LUFS")
        except Exception as exc:
            log.warning("Enhancement failed for '%s': %s — using raw WAV", song_name, exc)
            enhance_info = {"skipped": True, "error": str(exc)}
            demucs_input = wav_path
    else:
        enhance_info = {"skipped": True}

    # ── 3. Demucs separation ──────────────────────────────────────────────
    python_exe = _venv_python()
    hw_device = _detect_device() if device == "auto" else device
    _prog(0.20, f"Demucs ({model}) starting on {hw_device.upper()}… "
                 f"{'⚡ GPU accelerated' if hw_device != 'cpu' else '🐢 CPU mode'}")
    log.info("[stems] Using Python: %s", python_exe)
    log.info("[stems] Device: %s", hw_device)

    produced: Dict[str, Path] = {}

    try:
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [
                python_exe, "-m", "demucs",
                "-n", model,
                "-d", hw_device,
                "-o", tmp,
                str(demucs_input),
            ]
            log.info("[stems] Running: %s", " ".join(cmd))
            _prog(0.22, "Demucs running… (vocals / drums / bass / other)")

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                err_tail = (proc.stderr or "")[-500:]
                raise RuntimeError(
                    f"Demucs exited with code {proc.returncode}.\n"
                    f"stderr (last 500 chars):\n{err_tail}"
                )

            # ── Locate output directory ───────────────────────────────────
            # Demucs writes to: <tmp>/<model>/<input_stem>/
            stem_src_dir = Path(tmp) / model / demucs_input.stem
            if not stem_src_dir.exists():
                # Some versions (htdemucs_ft, mdx) use a different sub-folder
                matches = list(Path(tmp).rglob("vocals.wav"))
                if matches:
                    stem_src_dir = matches[0].parent
                    log.info("[stems] Found stems at: %s", stem_src_dir)
                else:
                    raise FileNotFoundError(
                        f"Demucs ran successfully but no stems found under {tmp}/. "
                        f"Searched for {model}/{demucs_input.stem}/"
                    )

            # ── Move stems into library/<song>/ ──────────────────────────
            _prog(0.80, "Moving stems into library…")
            for stem_name in ("vocals", "drums", "bass", "other"):
                src = stem_src_dir / f"{stem_name}.wav"
                if src.exists():
                    dst = dest_dir / f"{stem_name}.wav"
                    shutil.move(str(src), str(dst))
                    produced[stem_name] = dst
                    log.info("[stems] ✓ %s.wav", stem_name)

    except Exception as exc:
        return StemResult(song=song_name, success=False,
                          model=model,
                          enhance_info=enhance_info,
                          error=str(exc))

    if not produced:
        return StemResult(song=song_name, success=False,
                          model=model,
                          enhance_info=enhance_info,
                          error="Demucs ran but produced no stem files.")

    # ── 4. Per-stem LUFS normalisation ───────────────────────────────────
    _prog(0.86, "Normalising stem levels…")
    stem_info: Dict[str, dict] = {}

    try:
        from scripts.core.audio_enhance import enhance_stems as _enhance_stems

        stem_arrays: Dict[str, np.ndarray] = {}
        for sname, spath in produced.items():
            arr, _ = librosa.load(str(spath), sr=sr, mono=True)
            stem_arrays[sname] = arr

        normalised = _enhance_stems(stem_arrays, sr)

        for sname, arr in normalised.items():
            out_path = produced[sname]
            sf.write(str(out_path), arr, sr, subtype="PCM_24")
            rms_db = float(20 * np.log10(max(float(np.sqrt(np.mean(arr ** 2))), 1e-9)))
            stem_info[sname] = {
                "path":   str(out_path),
                "rms_db": round(rms_db, 1),
            }
            log.info("[stems] ✓ normalised %s  rms=%.1f dB", sname, rms_db)

    except Exception as exc:
        log.warning("[stems] Stem normalisation failed: %s", exc)
        for sname, spath in produced.items():
            stem_info[sname] = {"path": str(spath)}

    _prog(0.99, f"Done — {len(produced)} stems ready ✓")

    return StemResult(
        song         = song_name,
        success      = True,
        stems        = produced,
        stem_info    = stem_info,
        enhance_info = enhance_info,
        model        = model,
    )


# ---------------------------------------------------------------------------
# Batch helper (used by task_batch_stem_split)
# ---------------------------------------------------------------------------

def separate_batch(
    songs: List[str],
    enhance: bool        = True,
    model: str           = "htdemucs",
    skip_existing: bool  = True,
    progress_cb=None,    # callable(song_idx, total, song_name, stem_result)
) -> Dict[str, StemResult]:
    """
    Run separate_song_stems() on a list of songs.
    Returns dict of {song_name: StemResult}.
    """
    from scripts.core.paths import song_dir as _song_dir

    results: Dict[str, StemResult] = {}

    for i, song in enumerate(songs):
        # Skip songs that already have stems
        if skip_existing and (_song_dir(song) / "vocals.wav").exists():
            log.info("[batch] Skipping (stems exist): %s", song)
            results[song] = StemResult(
                song=song, success=True,
                stems={s: _song_dir(song) / f"{s}.wav"
                       for s in ("vocals", "drums", "bass", "other")
                       if (_song_dir(song) / f"{s}.wav").exists()},
                stem_info={}, enhance_info={"skipped": True},
            )
            if progress_cb:
                progress_cb(i, len(songs), song, results[song])
            continue

        def _cb(prog: float, msg: str, _i=i, _total=len(songs)) -> None:
            # Scale individual-song progress into the batch slot
            batch_prog = (_i + prog) / _total
            if progress_cb:
                progress_cb(_i, _total, song, None, batch_prog, msg)

        log.info("[batch] [%d/%d] Splitting: %s", i + 1, len(songs), song)
        result = separate_song_stems(song, enhance=enhance, model=model, progress_cb=_cb)
        results[song] = result

        if progress_cb:
            progress_cb(i, len(songs), song, result)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s  %(message)s")

    ap = argparse.ArgumentParser(
        description="Separate a library song into Demucs stems."
    )
    ap.add_argument("song", help="Song folder name (must exist in library/)")
    ap.add_argument("--model", default="htdemucs",
                    help="Demucs model (htdemucs | htdemucs_ft | mdx_extra)")
    ap.add_argument("--no-enhance", dest="enhance", action="store_false",
                    help="Skip audio enhancement before split")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show which Python/Demucs would be used, then exit")
    args = ap.parse_args()

    if args.dry_run:
        print(f"Python  : {_venv_python()}")
        print(f"Demucs  : {shutil.which('demucs') or 'not on PATH'}")
        sys.exit(0)

    result = separate_song_stems(
        args.song,
        enhance=args.enhance,
        model=args.model,
        progress_cb=lambda p, m: print(f"  [{int(p*100):3d}%] {m}"),
    )

    if result.success:
        print(f"\n✅  Stems saved for: {args.song}")
        for name, info in result.stem_info.items():
            rms = info.get("rms_db", "?")
            print(f"   {name:<8}  {rms} dB RMS  → {info.get('path', '?')}")
    else:
        print(f"\n❌  Failed: {result.error}")
        sys.exit(1)
