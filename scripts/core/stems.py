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
    preview_seconds: Optional[float] = None,
    output_dir: Optional[Path]       = None,
) -> StemResult:
    """
    Enhance and split a single library song into vocal / drum / bass / other stems.

    Parameters
    ----------
    song_name       : exact folder name inside library/
    enhance         : run audio-enhancement chain before Demucs (recommended)
    model           : Demucs model name  (htdemucs | htdemucs_ft | mdx_extra)
    device          : 'auto' (detect GPU), 'mps', 'cuda', or 'cpu'
    progress_cb     : optional callable(progress: float, message: str)
    preview_seconds : DJ_ENGINE_IMPROVEMENT_PLAN.md §2.2 — when set, only the
        first `preview_seconds` of audio are separated, cutting Demucs
        wall-clock roughly linearly. For interactive contexts (live preview,
        PiP widget) where a user is waiting on the result — NOT for
        producing a song's permanent library stems, which always want the
        full track. Requires `output_dir` (see below).
    output_dir      : where to write stem files. Defaults to library/<song>/
        (the permanent location) when `preview_seconds` is None — full
        parity with prior behaviour. Required when `preview_seconds` is set,
        specifically so a truncated preview separation can never land in
        and silently clobber a song's real library/<song>/{stem}.wav files;
        pass e.g. a tempfile.TemporaryDirectory() path.

    Returns
    -------
    StemResult with .success, .stems, .enhance_info, .error

    Raises
    ------
    ValueError
        If `preview_seconds` is set without an explicit `output_dir`.
    """
    import numpy as np
    import soundfile as sf

    try:
        import librosa
    except ImportError as exc:
        return StemResult(song=song_name, success=False,
                          error=f"librosa not available: {exc}")

    from scripts.core.paths import song_dir as _song_dir

    if preview_seconds is not None and output_dir is None:
        raise ValueError(
            "preview_seconds requires an explicit output_dir — a preview "
            "separation must never be written into library/<song>/, which "
            "would silently truncate that song's permanent stems."
        )

    def _prog(p: float, msg: str) -> None:
        log.info("[stems] %.0f%%  %s", p * 100, msg)
        if progress_cb:
            try:
                progress_cb(p, msg)
            except Exception:
                pass

    from scripts.core.audio_source import resolve_source_file

    dest_dir  = _song_dir(song_name)          # permanent location — read-only below
    write_dir = Path(output_dir) if output_dir is not None else dest_dir
    write_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the best available source, same priority order the render/
    # analysis path already uses (scripts/core/audio_source.py): full.wav
    # first, falling back to full_enhanced.wav. A song can end up with ONLY
    # full_enhanced.wav and no stems — e.g. Demucs failed *after* the
    # enhancement step had already written its output, or full.wav was later
    # reclaimed by storage/eviction before separation ever completed. Without
    # this fallback such a song can never be re-split again: the enhanced
    # mix is a perfectly good, full-length Demucs input already sitting on
    # disk, but the old full.wav-only check would reject it and force a
    # re-download from source to recover.
    source_path = resolve_source_file(song_name)
    if source_path is None:
        return StemResult(song=song_name, success=False,
                          error=f"No source audio (full.wav or full_enhanced.wav) "
                                f"found in library/{song_name}/")
    wav_path = source_path
    reused_enhanced = source_path.name == "full_enhanced.wav"

    # ── 1. Load audio ────────────────────────────────────────────────────
    _prog(0.03, f"Loading {song_name[:40]}…")
    sr = 44100
    try:
        audio, _ = librosa.load(str(wav_path), sr=sr, mono=True)
    except Exception as exc:
        return StemResult(song=song_name, success=False,
                          error=f"Could not load audio: {exc}")

    if preview_seconds is not None:
        max_samples = int(preview_seconds * sr)
        if len(audio) > max_samples:
            log.info("[stems] Preview mode: truncating %.1fs → %.1fs before separation",
                     len(audio) / sr, preview_seconds)
            audio = audio[:max_samples]

    # Preview mode's truncated `audio` only reaches Demucs if something on
    # disk actually reflects it — the enhance=True path below always writes
    # a fresh file built from `audio`, but enhance=False (and the "enhancement
    # itself failed" fallback) previously pointed straight at wav_path, the
    # ORIGINAL untruncated file, silently defeating the whole point of
    # preview_seconds. _preview_input() writes the truncated audio once and
    # is the fallback target in both of those cases instead of wav_path.
    _preview_input_path: Optional[Path] = None

    def _preview_input() -> Path:
        nonlocal _preview_input_path
        if _preview_input_path is None:
            _preview_input_path = write_dir / "preview_input.wav"
            sf.write(str(_preview_input_path), audio, sr, subtype="PCM_24")
        return _preview_input_path

    # ── 2. Enhancement chain ──────────────────────────────────────────────
    demucs_input = _preview_input() if preview_seconds is not None else wav_path
    enhance_info: dict = {}

    if enhance and reused_enhanced:
        # source_path is already a full_enhanced.wav from a prior attempt —
        # re-running the enhancement chain on already-enhanced audio would
        # double-apply LUFS normalisation/compression. Feed it to Demucs as
        # produced (preview_seconds truncation above still applies).
        enhance_info = {"reused_existing_enhanced_file": True}
    elif enhance:
        _prog(0.08, "Enhancing audio (HP · compress · air EQ · LUFS −14)…")
        try:
            from scripts.core.audio_enhance import EnhanceOptions, enhance_audio
            opts = EnhanceOptions(
                hp_filter      = True,
                lp_filter      = True,
                noise_gate     = True,
                compression    = True,
                air_eq         = True,
                lufs_target    = -14.0,
                true_peak_ceil = -1.0,
            )
            enh_audio, enh_report = enhance_audio(audio, sr, opts)   # audio already truncated above
            enh_path = write_dir / "full_enhanced.wav"
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
            demucs_input = _preview_input() if preview_seconds is not None else wav_path
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
            def _run_demucs(dev: str) -> subprocess.CompletedProcess:
                cmd = [
                    python_exe, "-m", "demucs",
                    "-n", model,
                    "-d", dev,
                    "-o", tmp,
                    str(demucs_input),
                ]
                log.info("[stems] Running: %s", " ".join(cmd))
                return subprocess.run(cmd, capture_output=True, text=True)

            _prog(0.22, "Demucs running… (vocals / drums / bass / other)")
            proc = _run_demucs(hw_device)

            # GPU backends (MPS/CUDA) can fail on some torch/driver combos —
            # retry once on CPU before giving up.
            if proc.returncode != 0 and hw_device != "cpu":
                log.warning(
                    "[stems] Demucs failed on %s (code %d) — retrying on CPU. stderr tail: %s",
                    hw_device, proc.returncode, (proc.stderr or "")[-300:],
                )
                _prog(0.25, f"Demucs failed on {hw_device.upper()} — retrying on CPU (slower)…")
                proc = _run_demucs("cpu")

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

            # ── Move stems into write_dir (library/<song>/, unless this is a
            #    preview separation — see output_dir above) ────────────────
            _prog(0.80, "Moving stems…" if write_dir != dest_dir else "Moving stems into library…")
            for stem_name in ("vocals", "drums", "bass", "other"):
                src = stem_src_dir / f"{stem_name}.wav"
                if src.exists():
                    dst = write_dir / f"{stem_name}.wav"
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
# Batch helper — NOT used by task_batch_stem_split (that function has its
# own independent loop in scripts/api/task_modules/stems.py, with its own
# skip_existing/all-4-stems check and its own _prune_after_split() call).
# Nothing in this codebase currently calls separate_batch(); kept as a
# library-level convenience for scripting, but its skip_existing check below
# only looks at vocals.wav — the same "1/4 stems counts as done" bug fixed
# in task_batch_stem_split's own loop — if this ever gets wired up for real,
# fix that first.
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
    import time

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
    ap.add_argument("--preview-seconds", type=float, default=None,
                    help="DJ_ENGINE_IMPROVEMENT_PLAN.md §2.2 — only separate the first N "
                         "seconds (preview-tier). Writes to a temp dir, never into library/.")
    ap.add_argument("--benchmark", action="store_true",
                    help="Run full-quality AND --preview-seconds (default 60s if not given) "
                         "back-to-back on this song and print wall-clock time for each. This "
                         "is the real benchmark for §2.2 — run it on your own machine (with "
                         "the actual MPS/CUDA device this engine uses), not in a generic "
                         "sandbox, since device acceleration is the whole point of the timing.")
    args = ap.parse_args()

    if args.dry_run:
        print(f"Python  : {_venv_python()}")
        print(f"Demucs  : {shutil.which('demucs') or 'not on PATH'}")
        sys.exit(0)

    if args.benchmark:
        preview_s = args.preview_seconds or 60.0

        t0 = time.monotonic()
        full_result = separate_song_stems(args.song, enhance=args.enhance, model=args.model)
        t_full = time.monotonic() - t0

        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.monotonic()
            preview_result = separate_song_stems(
                args.song, enhance=args.enhance, model=args.model,
                preview_seconds=preview_s, output_dir=Path(tmp),
            )
            t_preview = time.monotonic() - t0

        print(f"\n📊  Benchmark — {args.song}")
        print(f"   full  ({'ok' if full_result.success else 'FAILED'}):    {t_full:6.1f}s")
        print(f"   preview {preview_s:.0f}s ({'ok' if preview_result.success else 'FAILED'}): {t_preview:6.1f}s")
        if full_result.success and preview_result.success and t_full > 0:
            print(f"   speedup: {t_full / max(t_preview, 1e-6):.2f}×")
        sys.exit(0 if (full_result.success and preview_result.success) else 1)

    # Unlike --benchmark's tempfile.TemporaryDirectory(), this dir is NOT
    # auto-cleaned — the whole point of --preview-seconds on the CLI is to
    # leave inspectable output behind after the process exits, so deleting
    # it here would defeat that. It does mean repeated invocations
    # accumulate stems_preview_* dirs in the OS temp dir with nothing to
    # remove them; the path is printed below so it's at least visible and
    # the caller can clean up manually (`rm -rf <path>`).
    preview_output_dir = (
        Path(tempfile.mkdtemp(prefix="stems_preview_"))
        if args.preview_seconds is not None else None
    )

    result = separate_song_stems(
        args.song,
        enhance=args.enhance,
        model=args.model,
        preview_seconds=args.preview_seconds,
        output_dir=preview_output_dir,
    )

    if result.success:
        print(f"\n✅  Stems saved for: {args.song}")
        for name, info in result.stem_info.items():
            rms = info.get("rms_db", "?")
            print(f"   {name:<8}  {rms} dB RMS  → {info.get('path', '?')}")
        if preview_output_dir is not None:
            print(f"\n📁  Preview output dir (not auto-cleaned — remove manually "
                  f"when done): {preview_output_dir}")
    else:
        print(f"\n❌  Failed: {result.error}")
        sys.exit(1)
