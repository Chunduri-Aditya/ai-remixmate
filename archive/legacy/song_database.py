#!/usr/bin/env python3
# scripts/song_database.py
"""
Add a single song to the feature database.
Extracts tempo, chroma, and MFCC features using core.features
and stores them in models/song_embeddings.json.
"""

from __future__ import annotations
import argparse
from pathlib import Path

from scripts.core.paths import embeddings_path
from scripts.core.features import extract_features_from_wav, load_embeddings, save_embeddings


def add_song_to_db(song_path: Path, song_name: str | None = None) -> None:
    """Extract features and add a song entry to the DB."""
    if not song_path.exists():
        print(f"❌ File not found: {song_path}")
        return

    if song_path.suffix.lower() != ".wav":
        print("⚠️ Only .wav files are supported.")
        return

    song_name = song_name or song_path.stem

    print(f"🎧 Extracting features for: {song_name}")
    feats = extract_features_from_wav(song_path)

    db = load_embeddings()
    db[song_name] = feats.to_dict()
    save_embeddings(db)

    print(f"✅ Added '{song_name}' to database.")
    print(f"📦 DB path: {embeddings_path()}")


def main():
    ap = argparse.ArgumentParser(description="Add a single .wav file to the feature database.")
    ap.add_argument("--wav", required=True, help="Path to the song .wav file")
    ap.add_argument("--name", help="Optional custom song name (default: file stem)")
    args = ap.parse_args()

    add_song_to_db(Path(args.wav), args.name)


if __name__ == "__main__":
    main()