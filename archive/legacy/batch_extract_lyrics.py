#!/usr/bin/env python3
# scripts/batch_extract_lyrics.py

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import timedelta

from scripts.core.paths import SEPARATED

def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return str(timedelta(seconds=s))

def transcribe_song(model, song_dir: Path, language: str | None) -> bool:
    vocals = song_dir / "vocals.wav"
    out_txt = song_dir / "lyrics.txt"

    if not vocals.exists():
        print(f"⚠️ Skipping {song_dir.name} (no vocals.wav)")
        return False

    print(f"🎤 Transcribing: {song_dir.name}")
    try:
        # Whisper accepts language=None for auto-detect
        result = model.transcribe(str(vocals), language=language)
        segments = result.get("segments", [])

        if not segments:
            text = result.get("text", "").strip()
            if not text:
                print(f"⚠️ No speech detected for {song_dir.name}")
                return False
            out_txt.write_text(text + "\n", encoding="utf-8")
            print(f"✅ Saved (no segments): {out_txt}")
            return True

        # Save timestamped transcript
        with out_txt.open("w", encoding="utf-8") as f:
            for seg in segments:
                start = hhmmss(seg.get("start", 0))
                end = hhmmss(seg.get("end", 0))
                line = f"[{start} - {end}] {seg.get('text','').strip()}"
                f.write(line + "\n")

        print(f"✅ Saved: {out_txt}")
        return True

    except Exception as e:
        print(f"❌ Error with {song_dir.name}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium", help="Whisper model size (tiny, base, small, medium, large)")
    ap.add_argument("--force", action="store_true", help="Re-transcribe even if lyrics.txt exists")
    ap.add_argument("--language", default=None, help="Force language code (e.g., 'en'); default: auto-detect")
    args = ap.parse_args()

    try:
        import whisper
    except Exception:
        print("❌ Missing dependency. Install with: pip install -U openai-whisper")
        return

    print(f"🔁 Loading Whisper model ({args.model})...")
    model = whisper.load_model(args.model)

    if not SEPARATED.exists():
        print(f"⚠️ No stems directory found at {SEPARATED}")
        return

    song_dirs = sorted([p for p in SEPARATED.iterdir() if p.is_dir()])
    if not song_dirs:
        print(f"⚠️ No song folders found in {SEPARATED}")
        return

    done = 0
    for sd in song_dirs:
        out_txt = sd / "lyrics.txt"
        if out_txt.exists() and not args.force:
            print(f"⏭️  Skipping (already transcribed): {sd.name}")
            continue
        if transcribe_song(model, sd, language=args.language):
            done += 1

    print(f"\n✅ Completed. New/updated transcripts: {done}")

if __name__ == "__main__":
    main()