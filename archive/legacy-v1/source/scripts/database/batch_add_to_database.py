import json
from pathlib import Path

import librosa
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_FOLDER = PROJECT_ROOT / "audio_input"
DB_PATH = PROJECT_ROOT / "models" / "song_embeddings.json"
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".aiff")


def extract_features(filepath):
    """Extract audio features for database."""
    try:
        y, sr = librosa.load(str(filepath), sr=44100, duration=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return {
            "tempo": float(tempo),
            "chroma": np.mean(chroma, axis=1).tolist(),
            "mfcc": np.mean(mfcc, axis=1).tolist(),
            "path": str(filepath),
        }
    except Exception as e:
        print(f"   ⚠️  Error extracting features: {e}")
        return None


def load_database(db_path=DB_PATH):
    """Load existing database."""
    db_path = Path(db_path)
    if db_path.exists():
        try:
            with db_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading database: {e}")
            return {}
    return {}


def save_database(db, db_path=DB_PATH):
    """Save database to file."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.open("w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


def find_audio_files(root_dir):
    """Find all audio files recursively."""
    root_dir = Path(root_dir)
    if not root_dir.exists():
        return []

    return sorted(
        path for path in root_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    )


def batch_add(audio_folder=AUDIO_FOLDER, db_path=DB_PATH):
    """Add all audio files from audio_input folder to database."""
    audio_folder = Path(audio_folder)
    print("=" * 60)
    print("🎵 Adding Songs to Database")
    print("=" * 60)

    if not audio_folder.exists():
        print(f"❌ Audio folder not found: {audio_folder}")
        return

    db = load_database(db_path)
    audio_files = find_audio_files(audio_folder)

    if not audio_files:
        print(f"⚠️  No audio files found in {audio_folder}")
        return

    print(f"\n📁 Found {len(audio_files)} audio file(s)")
    print(f"📊 Database currently has {len(db)} song(s)\n")

    added = 0
    skipped = 0
    failed = 0

    for filepath in audio_files:
        song_name = filepath.stem

        if song_name in db:
            print(f"⏩ Skipping (already in DB): {song_name}")
            skipped += 1
            continue

        print(f"🎧 Processing: {song_name}")
        print(f"   📂 {filepath}")

        features = extract_features(filepath)
        if features is None:
            print("   ❌ Failed to extract features")
            failed += 1
            continue

        db[song_name] = features
        added += 1
        print("   ✅ Added to database")

    save_database(db, db_path)

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Added: {added} new song(s)")
    print(f"⏩ Skipped: {skipped} song(s) (already in DB)")
    print(f"❌ Failed: {failed} song(s)")
    print(f"📊 Total in database: {len(db)} song(s)")
    print(f"💾 Database saved to: {db_path}")


if __name__ == "__main__":
    batch_add()
