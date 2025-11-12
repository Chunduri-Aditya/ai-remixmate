# scripts/song_database.py

import os
import librosa
import numpy as np
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "../models/song_embeddings.json")

def extract_features(filepath):
    y, sr = librosa.load(filepath)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "tempo": float(tempo),
        "chroma": np.mean(chroma, axis=1).tolist(),
        "mfcc": np.mean(mfcc, axis=1).tolist()
    }

def add_song_to_db(song_path, song_name):
    features = extract_features(song_path)

    # Load existing DB
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            db = json.load(f)
    else:
        db = {}

    db[song_name] = features

    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

    print(f"✅ Added '{song_name}' to DB.")

if __name__ == "__main__":
    song_path = input("Enter path to .wav file: ")
    song_name = input("Enter name (or title) for this song: ")
    add_song_to_db(song_path, song_name)