# scripts/recommend_match.py

import json
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../models/song_embeddings.json")

def extract_features(filepath):
    y, sr = librosa.load(filepath)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "chroma": np.mean(chroma, axis=1),
        "mfcc": np.mean(mfcc, axis=1)
    }

def recommend_similar_song(song_path):
    # Load song DB
    with open(DB_PATH, "r") as f:
        db = json.load(f)

    # Extract features from input song
    input_features = extract_features(song_path)
    input_vec = np.concatenate([input_features["mfcc"], input_features["chroma"]]).reshape(1, -1)

    similarities = []

    for title, data in db.items():
        db_vec = np.concatenate([data["mfcc"], data["chroma"]]).reshape(1, -1)
        similarity = cosine_similarity(input_vec, db_vec)[0][0]
        similarities.append((title, similarity))

    # Sort by highest similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"\n🎯 Best Match: {similarities[0][0]}")
    print(f"   Similarity Score: {similarities[0][1]:.4f}\n")

    print("📊 Top Matches:")
    for title, score in similarities:
        print(f"- {title:30} | Similarity: {score:.4f}")

if __name__ == "__main__":
    song_path = input("Enter path to song (.wav): ")
    recommend_similar_song(song_path)