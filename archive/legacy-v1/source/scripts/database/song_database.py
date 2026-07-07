import json
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "models" / "song_embeddings.json"


def extract_features(filepath):
    y, sr = librosa.load(str(filepath))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    return {
        "tempo": float(tempo),
        "chroma": np.mean(chroma, axis=1).tolist(),
        "mfcc": np.mean(mfcc, axis=1).tolist(),
    }


def add_song_to_db(song_path, song_name, db_path=DB_PATH):
    db_path = Path(db_path)
    features = extract_features(song_path)

    if db_path.exists():
        with db_path.open("r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    db[song_name] = features

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)

    print(f"✅ Added '{song_name}' to DB.")


if __name__ == "__main__":
    song_path = input("Enter path to .wav file: ")
    song_name = input("Enter name (or title) for this song: ")
    add_song_to_db(song_path, song_name)
