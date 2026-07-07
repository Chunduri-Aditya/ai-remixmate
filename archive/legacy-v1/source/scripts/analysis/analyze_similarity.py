from functools import lru_cache
from pathlib import Path

import numpy as np
import librosa
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STEMS_DIR = PROJECT_ROOT / "separated" / "htdemucs"
MAX_CANDIDATES = 10  # Limit to top 10 tempo-matching songs

@lru_cache(maxsize=1)
def _get_similarity_model():
    try:
        from sentence_transformers import SentenceTransformer, util
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional analysis dependency. Install it with: "
            "python -m pip install -e '.[analysis]'"
        ) from exc

    return SentenceTransformer("all-MiniLM-L6-v2"), util

def get_audio_features(path):
    y, sr = librosa.load(str(path), sr=None, mono=True)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc, axis=1), float(tempo)  # Cast tempo to float

def get_lyrics_embedding(song_name, stems_dir=STEMS_DIR):
    path = Path(stems_dir) / song_name / "lyrics.txt"
    if not path.exists():
        print(f"⚠️ Missing lyrics: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        text = f.read()
    model, _ = _get_similarity_model()
    return model.encode(text, convert_to_tensor=True)

def get_similarity(song1, song2, emb1, emb2, stems_dir=STEMS_DIR):
    if emb1 is None or emb2 is None:
        return -1
    _, util = _get_similarity_model()
    lyr_score = util.pytorch_cos_sim(emb1, emb2).item()

    stems_dir = Path(stems_dir)
    feat1, tempo1 = get_audio_features(stems_dir / song1 / "other.wav")
    feat2, tempo2 = get_audio_features(stems_dir / song2 / "other.wav")
    audio_score = 1 - np.linalg.norm(feat1 - feat2) / 100  # scaled
    tempo_score = 1 - abs(tempo1 - tempo2) / max(tempo1, tempo2, 1.0)

    total_score = 0.5 * lyr_score + 0.3 * audio_score + 0.2 * tempo_score
    return total_score

def find_best_match(base_song, top_k=5, stems_dir=STEMS_DIR):
    stems_dir = Path(stems_dir)
    if not stems_dir.exists():
        print(f"❌ Separated stems folder not found: {stems_dir}")
        return []

    print(f"🎧 Analyzing: {base_song}")
    base_other = stems_dir / base_song / "other.wav"
    base_feat, base_tempo = get_audio_features(base_other)
    base_emb = get_lyrics_embedding(base_song)

    all_songs = [
        d.name for d in stems_dir.iterdir()
        if d.is_dir() and d.name != base_song
    ]

    # Step 1: Tempo filtering
    tempo_diff = []
    for s in all_songs:
        try:
            _, t = get_audio_features(stems_dir / s / "other.wav")
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
        emb = get_lyrics_embedding(song, stems_dir=stems_dir)
        sim = get_similarity(base_song, song, base_emb, emb, stems_dir=stems_dir)
        if sim > 0:
            results.append((song, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find separated songs with similar lyrics/audio features.")
    parser.add_argument("base_song", nargs="?", default="Antidote", help="Folder name under separated/htdemucs.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of matches to print.")
    args = parser.parse_args()

    matches = find_best_match(args.base_song, top_k=args.top_k)
    print(f"\n🎯 Top matches for: {args.base_song}\n")
    for i, (song, score) in enumerate(matches):
        print(f"{i+1}. {song:<40} | Similarity: {score:.4f}")
