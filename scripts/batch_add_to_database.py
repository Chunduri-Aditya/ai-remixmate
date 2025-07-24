# scripts/batch_add_to_database.py

import os
import json
import librosa
import numpy as np

AUDIO_FOLDER = "../audio_input"
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

def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_database(db):
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def batch_add():
    db = load_database()
    added = 0

    for filename in os.listdir(AUDIO_FOLDER):
        if not filename.endswith(".wav"):
            continue

        song_name = os.path.splitext(filename)[0]

        if song_name in db:
            print(f"⏩ Skipping (already in DB): {song_name}")
            continue

        print(f"🎧 Adding to DB: {song_name}")
        filepath = os.path.join(AUDIO_FOLDER, filename)
        features = extract_features(filepath)
        db[song_name] = features
        added += 1

    save_database(db)
    print(f"\n✅ Added {added} new song(s) to database.")

if __name__ == "__main__":
    batch_add()