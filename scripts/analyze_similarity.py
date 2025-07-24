# analyze_similarity.py
import os
import numpy as np
import librosa
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

STEMS_DIR = "./separated/htdemucs"
MAX_CANDIDATES = 10  # Limit to top 10 tempo-matching songs

# Load sentence-transformer for lyrics similarity
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_audio_features(path):
    y, sr = librosa.load(path, sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1), float(tempo)  # Cast tempo to float

def get_lyrics_embedding(song_name):
    path = os.path.join(STEMS_DIR, song_name, "lyrics.txt")
    if not os.path.exists(path):
        print(f"⚠️ Missing lyrics: {path}")
        return None
    with open(path, "r") as f:
        text = f.read()
    return model.encode(text, convert_to_tensor=True)

def get_similarity(song1, song2, emb1, emb2):
    if emb1 is None or emb2 is None:
        return -1
    lyr_score = util.pytorch_cos_sim(emb1, emb2).item()

    feat1, tempo1 = get_audio_features(os.path.join(STEMS_DIR, song1, "other.wav"))
    feat2, tempo2 = get_audio_features(os.path.join(STEMS_DIR, song2, "other.wav"))
    audio_score = 1 - np.linalg.norm(feat1 - feat2) / 100  # scaled
    tempo_score = 1 - abs(tempo1 - tempo2) / max(tempo1, tempo2)

    total_score = 0.5 * lyr_score + 0.3 * audio_score + 0.2 * tempo_score
    return total_score

def find_best_match(base_song, top_k=5):
    print(f"🎧 Analyzing: {base_song}")
    base_other = os.path.join(STEMS_DIR, base_song, "other.wav")
    base_feat, base_tempo = get_audio_features(base_other)
    base_emb = get_lyrics_embedding(base_song)

    all_songs = [
        d for d in os.listdir(STEMS_DIR)
        if os.path.isdir(os.path.join(STEMS_DIR, d)) and d != base_song
    ]

    # Step 1: Tempo filtering
    tempo_diff = []
    for s in all_songs:
        try:
            _, t = get_audio_features(os.path.join(STEMS_DIR, s, "other.wav"))
            tempo_val = float(t)
            diff = abs(tempo_val - base_tempo)
            tempo_diff.append((s, diff))
        except Exception as e:
            print(f"⚠️ Skipping {s} due to audio error: {e}")
            continue

    # Show preview
    print(f"\n🔎 Top {MAX_CANDIDATES} tempo-close songs to {base_song}:\n")
    tempo_filtered = sorted(tempo_diff, key=lambda x: x[1])[:MAX_CANDIDATES]
    for s, d in tempo_filtered:
        print(f" - {s:<35} | Tempo diff: {d:.2f}")

    candidates = [s for s, _ in tempo_filtered]

    # Step 2: Full similarity check with progress bar
    results = []
    for song in tqdm(candidates, desc="🔍 Comparing songs"):
        emb = get_lyrics_embedding(song)
        sim = get_similarity(base_song, song, base_emb, emb)
        if sim > 0:
            results.append((song, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "Antidote"
    matches = find_best_match(base)
    print(f"\n🎯 Top matches for: {base}\n")
    for i, (song, score) in enumerate(matches):
        print(f"{i+1}. {song:<40} | Similarity: {score:.4f}")