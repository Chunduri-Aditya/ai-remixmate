"""Import a folder of your own music into the library.

Usage: python -m scripts.scan <folder> [--stems] [--analyze]
       ./start.sh scan <folder> [--stems] [--analyze]

Each audio file under <folder> (searched recursively; hidden files, dot
folders and anything inside the library itself are skipped) becomes
library/<name>/full.wav (mono 44.1 kHz, via ffmpeg). <name> is the file name
without its extension, normalized to NFC and passed through
sanitize_song_name, so non ASCII names such as "Café del Mar" survive.

  --stems    split each song into Demucs stems (skipped when stems exist)
  --analyze  run the song analysis (tempo, key, structure) on full.wav

Every song is registered in the library index with source="local". No
licence record is written: the file is your own, it is not a CC catalog
track, and library/ is never served in hosted mode.

Rescans are safe. A song whose full.wav or any stem already exists is
reported as "already in library" and is not converted again. Scan does not
compare contents, so a different file with the same name also reads as
already present.

Two files in one scan whose names collide (compared NFC and case folded,
as on macOS volumes) are never merged: the first in sorted path order wins
and the second is skipped with a message that names both files.

One bad file never stops the folder: a corrupt file, an empty name, a
collision or a failed stems, analysis or index step prints one line and the
scan moves on.

Exit codes:
  0  every audio file imported or already in the library
  1  no audio files found, or one or more files skipped or failed
  2  the folder does not exist, or ffmpeg is not on PATH
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

# Modules, not names: paths.LIBRARY_DIR and get_library_manager are read at
# call time, so a test or a caller that redirects them is honoured. Nothing
# from the API package is imported: importing its job store marks live API
# jobs FAILED (OPS-08; tests/test_scan.py::test_scan_never_imports_the_job_store).
from scripts.core import library as library_mod
from scripts.core import paths

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aiff", ".aif"}
STEM_NAMES = ("vocals", "drums", "bass", "other")


def song_name_for(src: Path) -> str:
    """Library name for a source file: NFC first, so a decomposed (NFD)
    accent is not stripped as a lone combining mark."""
    return paths.sanitize_song_name(unicodedata.normalize("NFC", src.stem))


def find_audio(folder: Path) -> list[Path]:
    """Audio files under folder, sorted, without dotfiles (AppleDouble
    "._x.mp3"), dot folders or files inside the library root."""
    library_root = paths.LIBRARY_DIR.resolve()
    found = []
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if not p.is_file() or p.suffix.lower() not in AUDIO_EXTS:
            continue
        if p.resolve().is_relative_to(library_root):
            continue
        found.append(p)
    return sorted(found)


def _has_stems(d: Path) -> bool:
    return any((d / f"{s}.{ext}").exists() for s in STEM_NAMES for ext in ("flac", "wav"))


def _present(d: Path) -> bool:
    return (d / "full.wav").exists() or _has_stems(d)


def import_file(src: Path, name: str) -> Path:
    """Convert src to library/<name>/full.wav and return that path.

    ffmpeg writes a temp file in the song dir that replaces full.wav only on
    success, so a corrupt source never leaves a partial full.wav behind. On
    failure the temp file and any dir this call created are removed and a
    RuntimeError carries ffmpeg's last stderr line."""
    dest = paths.song_dir(name)
    if dest.resolve().parent != paths.LIBRARY_DIR.resolve():
        raise ValueError(f"song name {name!r} resolves outside the library")
    created = not dest.exists()
    dest.mkdir(parents=True, exist_ok=True)
    wav = dest / "full.wav"
    tmp = dest / f".full.{os.getpid()}.tmp.wav"
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src.resolve()),
         "-vn", "-ac", "1", "-ar", "44100", str(tmp)],
        capture_output=True, text=True, errors="replace",
    )
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        if created and not any(dest.iterdir()):
            dest.rmdir()
        lines = proc.stderr.strip().splitlines()
        raise RuntimeError(lines[-1] if lines else f"ffmpeg exit code {proc.returncode}")
    os.replace(tmp, wav)
    return wav


def separate_stems(name: str) -> None:
    """Demucs stems for a library song, called as scripts/download.py does."""
    from scripts.core.stems import separate_song_stems

    result = separate_song_stems(name, enhance=False)
    if not result.success:
        raise RuntimeError(result.error or "stem separation failed")


def analyze(name: str) -> None:
    """Song analysis on full.wav; does not need stems."""
    from scripts.core.analysis_pipeline import run_song_analysis

    run_song_analysis(name)


def _has_analysis(d: Path) -> bool:
    from scripts.core.analysis_pipeline import has_analysis

    return has_analysis(d)


def _register(name: str, imported: bool, made_stems: bool) -> None:
    """Index a new import as source="local". An existing song is indexed only
    when the index lacks it, or refreshed when new stems landed, keeping its
    recorded source (register() rescans the dir and resets source)."""
    mgr = library_mod.get_library_manager()
    entry = mgr.song_info(name)
    if imported:
        mgr.register(name, source="local")
    elif entry is None:
        # Already on disk but unindexed: a licence record names the real
        # source (register() reads it when source is None); otherwise local.
        has_licence = (paths.song_dir(name) / "license.json").exists()
        mgr.register(name, source=None if has_licence else "local")
    elif made_stems:
        mgr.register(name, source=entry.source)


def _err(msg: str) -> None:
    print(msg, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="backslashreplace")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--stems", action="store_true", help="split each song into Demucs stems")
    ap.add_argument("--analyze", action="store_true", help="run the song analysis on full.wav")
    args = ap.parse_args(argv)
    folder: Path = args.folder

    if not folder.is_dir():
        _err(f"not a folder: {folder}")
        return 2
    if shutil.which("ffmpeg") is None:
        _err("ffmpeg not found on PATH. Install it (macOS: brew install ffmpeg; "
             "Debian or Ubuntu: sudo apt install ffmpeg) and run the scan again.")
        return 2

    files = find_audio(folder)
    if not files:
        _err(f"no audio files found in {folder} (looked for {', '.join(sorted(AUDIO_EXTS))})")
        return 1

    paths.LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    imported = present = skipped = failed = partial = 0
    claimed: dict[str, Path] = {}

    for f in files:
        name = song_name_for(f)
        if not name:
            _err(f"skip {f}: no usable characters in the file name")
            skipped += 1
            continue
        key = name.casefold()
        if key in claimed:
            _err(f"skip {f}: its name {name!r} is already taken in this scan by {claimed[key]}")
            skipped += 1
            continue
        claimed[key] = f

        d = paths.song_dir(name)
        is_new = False
        if _present(d):
            print(f"already in library: {f} -> {name} "
                  "(a song with this name exists; contents not compared)")
            present += 1
        else:
            try:
                import_file(f, name)
            except Exception as exc:
                _err(f"failed {f}: {exc}")
                failed += 1
                continue
            print(f"imported {f} -> {name}")
            imported += 1
            is_new = True

        made_stems = False
        if args.stems and not _has_stems(d):
            try:
                separate_stems(name)
                made_stems = True
                print(f"stems done: {name}")
            except Exception as exc:
                _err(f"stems failed for {name}: {exc}")
                partial += 1

        if args.analyze:
            try:
                if not _has_analysis(d):
                    analyze(name)
                    print(f"analysis done: {name}")
            except Exception as exc:
                _err(f"analysis failed for {name}: {exc}")
                partial += 1

        try:
            _register(name, is_new, made_stems)
        except Exception as exc:
            _err(f"index registration failed for {name}: {exc}")
            partial += 1

    summary = (f"{imported} imported, {present} already in library, {skipped} skipped, "
               f"{failed} failed (of {len(files)} audio files) in {paths.LIBRARY_DIR}")
    if partial:
        summary += f"; {partial} stems, analysis or index step(s) failed"
    print(summary)
    return 1 if (skipped or failed or partial) else 0


if __name__ == "__main__":
    sys.exit(main())
