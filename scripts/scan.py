"""Import a folder of your own music into the library.

Usage: python -m scripts.scan <folder> [--stems] [--analyze]

Each audio file becomes library/<name>/full.wav (mono 44.1 kHz, via ffmpeg).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from scripts.core.paths import LIBRARY_DIR, ensure_directories, sanitize_song_name, song_dir

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aiff", ".aif"}


def find_audio(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)


def import_file(src: Path) -> Path | None:
    name = sanitize_song_name(src.stem)
    if not name:
        return None
    dest = song_dir(name)
    if dest.resolve().parent != LIBRARY_DIR.resolve():
        return None
    wav = dest / "full.wav"
    if wav.exists():
        return wav
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-ac", "1", "-ar", "44100", str(wav)],
        check=True,
    )
    return wav


def separate_stems(name: str) -> None:
    raise NotImplementedError("scan: stems")


def analyze(name: str) -> None:
    raise NotImplementedError("scan: analysis")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--stems", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args(argv)
    if not args.folder.is_dir():
        print(f"not a folder: {args.folder}", file=sys.stderr)
        return 2
    ensure_directories()
    files = find_audio(args.folder)
    done = 0
    for f in files:
        wav = import_file(f)
        if wav is None:
            print(f"skip {f.name}")
            continue
        if args.stems:
            separate_stems(wav.parent.name)
        if args.analyze:
            analyze(wav.parent.name)
        done += 1
        print(f"imported {f.name} -> {wav.parent.name}")
    print(f"{done} of {len(files)} files imported into {LIBRARY_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
