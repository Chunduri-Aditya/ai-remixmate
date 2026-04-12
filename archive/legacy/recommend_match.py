#!/usr/bin/env python3
# scripts/recommend_match.py
"""
Recommend the most similar songs from the feature DB for a given WAV.
Uses the same (chroma + MFCC) representation and cosine similarity as the rest of the project.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from scripts.core.paths import embeddings_path
from scripts.core.features import (
    extract_features_from_wav,
    load_embeddings,
    FeatureVector,
    cosine_similarity,
)

def _to_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else (v / n)

def recommend_similar_song(song_path: Path, top_k: int = 10) -> None:
    if not song_path.exists():
        print(f"❌ File not found: {song_path}")
        return

    db = load_embeddings()
    if not db:
        print(f"⚠️ Feature DB is empty or missing at: {embeddings_path()}")
        return

    # Query features
    q_feats = extract_features_from_wav(song_path)
    q_vec = _to_unit(np.concatenate([q_feats.chroma, q_feats.mfcc]).astype(np.float32))

    # Score against DB
    scored = []
    for name, d in db.items():
        try:
            fv = FeatureVector.from_dict(d)
            v = _to_unit(np.concatenate([fv.chroma, fv.mfcc]).astype(np.float32))
            s = cosine_similarity(q_vec, v)  # [-1, 1]
            scored.append((name, s))
        except Exception as e:
            print(f"⚠️ Skipping '{name}' (bad vector): {e}")

    if not scored:
        print("⚠️ No comparable entries found in DB.")
        return

    scored.sort(key=lambda x: x[1], reverse=True)

    best_name, best_score = scored[0]
    print(f"\n🎯 Best Match: {best_name}")
    print(f"   Cosine Similarity: {best_score:.4f}\n")

    print("📊 Top Matches:")
    for name, s in scored[:top_k]:
        print(f"- {name:30s} | similarity: {s:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Recommend similar songs from the DB for a given WAV.")
    ap.add_argument("--wav", required=True, help="Path to the query .wav file")
    ap.add_argument("--topk", type=int, default=10, help="How many matches to display")
    args = ap.parse_args()

    recommend_similar_song(Path(args.wav), top_k=args.topk)

if __name__ == "__main__":
    main()