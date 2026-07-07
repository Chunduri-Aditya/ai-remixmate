import argparse
import json
from pathlib import Path

import librosa
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "models" / "song_embeddings.json"


def extract_features(filepath):
    y, sr = librosa.load(str(filepath))
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "chroma": np.mean(chroma, axis=1),
        "mfcc": np.mean(mfcc, axis=1),
    }


def recommend_similar_song(song_path, db_path=DB_PATH):
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(
            f"Song database not found: {db_path}. "
            "Run python scripts/database/batch_add_to_database.py first."
        )

    with db_path.open("r", encoding="utf-8") as f:
        db = json.load(f)

    input_features = extract_features(song_path)
    input_vec = np.concatenate([input_features["mfcc"], input_features["chroma"]]).reshape(1, -1)

    similarities = []
    for title, data in db.items():
        db_vec = np.concatenate([data["mfcc"], data["chroma"]]).reshape(1, -1)
        similarity = cosine_similarity(input_vec, db_vec)[0][0]
        similarities.append((title, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🎯 Best Match: {similarities[0][0]}")
    print(f"   Similarity Score: {similarities[0][1]:.4f}\n")

    print("📊 Top Matches:")
    for title, score in similarities:
        print(f"- {title:30} | Similarity: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend similar songs from models/song_embeddings.json.")
    parser.add_argument("song_path", nargs="?", help="Path to an audio file.")
    args = parser.parse_args()

    song_path = args.song_path or input("Enter path to song (.wav): ")
    recommend_similar_song(song_path)
