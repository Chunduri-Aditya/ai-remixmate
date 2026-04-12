#!/usr/bin/env python3
# scripts/extract_lyrics.py
"""
Transcribe a single vocals.wav file into a timestamped lyrics.txt
using OpenAI Whisper (any supported model).
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta

from scripts.core.paths import SEPARATED

def hhmmss(seconds: float) -> str:
    """Convert seconds → HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))

def transcribe_song(vocals_path: Path, model_size: str = "medium", language: str | None = None) -> Path | None:
    """Transcribe a single vocals.wav file with Whisper and save lyrics.txt next to it."""
    if not vocals_path.exists():
        print(f"❌ File not found: {vocals_path}")
        return None

    try:
        import whisper
    except Exception:
        print("❌ Missing dependency. Install with: pip install -U openai-whisper")
        return None

    print(f"🔁 Loading Whisper model ({model_size})...")
    model = whisper.load_model(model_size)

    print(f"🎤 Transcribing: {vocals_path}")
    result = model.transcribe(str(vocals_path), verbose=False, language=language)

    segments = result.get("segments", [])
    if not segments:
        text = result.get("text", "").strip()
        if not text:
            print("⚠️ No speech detected.")
            return None
        lines = [text]
    else:
        lines = [f"[{hhmmss(seg['start'])} - {hhmmss(seg['end'])}] {seg['text'].strip()}" for seg in segments]

    song_dir = vocals_path.parent
    out_path = song_dir / "lyrics.txt"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Lyrics saved to: {out_path}")
    return out_path

def main():
    ap = argparse.ArgumentParser(description="Transcribe vocals.wav into lyrics.txt using Whisper.")
    ap.add_argument("--song", required=True, help="Song folder name under separated/htdemucs or full path to vocals.wav")
    ap.add_argument("--model", default="medium", help="Whisper model size (tiny, base, small, medium, large)")
    ap.add_argument("--language", default=None, help="Force language code (e.g., 'en'); default: auto-detect")
    args = ap.parse_args()

    # Accept either a folder name or full path
    song_path = Path(args.song)
    vocals_path = song_path if song_path.suffix == ".wav" else (SEPARATED / song_path / "vocals.wav")

    transcribe_song(vocals_path, model_size=args.model, language=args.language)

if __name__ == "__main__":
    main()