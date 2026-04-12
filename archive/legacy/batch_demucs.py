#!/usr/bin/env python3
# scripts/batch_demucs.py

from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path

from scripts.core.paths import AUDIO_IN, SEPARATED

DEFAULT_MODEL = "htdemucs"  # other options: mdx_extra, htdemucs_ft, etc.

def is_already_separated(song_stem: str) -> bool:
    """We consider a song separated if the 'other.wav' stem exists."""
    return (SEPARATED / song_stem / "other.wav").exists()

def separate_all(model: str = DEFAULT_MODEL, pattern: str = "*.wav") -> None:
    # Ensure base folders exist
    AUDIO_IN.mkdir(parents=True, exist_ok=True)
    SEPARATED.mkdir(parents=True, exist_ok=True)

    # Confirm demucs exists
    if shutil.which("demucs") is None:
        print("❌ Could not find 'demucs' on PATH. Install with: pip install demucs")
        return

    wavs = sorted(AUDIO_IN.glob(pattern))
    if not wavs:
        print(f"⚠️ No WAV files found in {AUDIO_IN} matching '{pattern}'.")
        return

    print(f"🎛️  Model: {model}")
    print(f"📂 Input: {AUDIO_IN}")
    print(f"📦 Output root: {SEPARATED}  (Demucs will create '{model}/<song>/' inside)")
    print()

    for wav in wavs:
        stem = wav.stem
        out_dir = SEPARATED / stem  # where our code expects stems after move by demucs into model dir
        # demucs will actually write to SEPARATED/<model>/<stem>/; our checks look for SEPARATED/<stem>/other.wav after demucs places it
        # but we check existence in the place we actually use:
        already = (SEPARATED / stem / "other.wav").exists() or (SEPARATED.parent / model / stem / "other.wav").exists()
        if is_already_separated(stem) or already:
            print(f"⏭️  Skipping already separated: {wav.name}")
            continue

        print(f"🎧 Separating: {wav.name}")
        cmd = [
            "demucs",
            "-n", model,
            "-o", str(SEPARATED.parent),  # demucs will place under SEPARATED.parent/<model>/<stem>/
            str(wav),
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ demucs failed for {wav.name}: {e}")
            continue

    print("\n✅ Separation pass complete.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Demucs model name (e.g., htdemucs, mdx_extra)")
    ap.add_argument("--glob", default="*.wav", help="Glob for input files in audio_input/ (e.g., '*.wav')")
    args = ap.parse_args()
    separate_all(model=args.model, pattern=args.glob)

if __name__ == "__main__":
    main()