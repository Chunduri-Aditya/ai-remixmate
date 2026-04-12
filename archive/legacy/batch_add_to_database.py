#!/usr/bin/env python3
# scripts/batch_add_to_database.py

from __future__ import annotations
import argparse
from pathlib import Path

from scripts.core.paths import AUDIO_IN, embeddings_path
from scripts.core.features import (
    extract_features_from_wav,
    load_embeddings,
    save_embeddings,
)

def batch_add(pattern: str = "*.wav") -> None:
    AUDIO_IN.mkdir(parents=True, exist_ok=True)

    db = load_embeddings()
    added = 0

    wavs = sorted(AUDIO_IN.glob(pattern))
    if not wavs:
        print(f"No audio files found in {AUDIO_IN} matching pattern '{pattern}'.")
        return

    for wav in wavs:
        song_name = wav.stem
        if song_name in db:
            print(f"⏩ Skipping (already in DB): {song_name}")
            continue

        print(f"🎧 Adding to DB: {song_name}")
        feats = extract_features_from_wav(wav)  # tempo, chroma (mean), mfcc (mean)
        db[song_name] = feats.to_dict()
        added += 1

    save_embeddings(db)
    print(f"\n✅ Added {added} new song(s) to database. Total: {len(db)}")
    print(f"📦 DB path: {embeddings_path()}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="*.wav",
                    help="Glob for files under audio_input/ (e.g., '*.wav' or 'set_*.wav')")
    args = ap.parse_args()
    batch_add(args.glob)

if __name__ == "__main__":
    main()