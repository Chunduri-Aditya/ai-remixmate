# scripts/batch_add_to_database.py

import os
import sys
import json
import librosa
import numpy as np
from pathlib import Path

# Add src to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
github_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up from scripts/database/ to github/
sys.path.insert(0, os.path.join(github_dir, "src"))

AUDIO_FOLDER = os.path.join(github_dir, "audio_input")
DB_PATH = os.path.join(github_dir, "models", "song_embeddings.json")

# Supported audio formats
AUDIO_EXTENSIONS = ('.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.aiff')

def extract_features(filepath):
    """Extract audio features for database."""
    try:
        y, sr = librosa.load(filepath, sr=44100, duration=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

        return {
            "tempo": float(tempo),
            "chroma": np.mean(chroma, axis=1).tolist(),
            "mfcc": np.mean(mfcc, axis=1).tolist(),
            "path": filepath  # Store path for reference
        }
    except Exception as e:
        print(f"   ⚠️  Error extracting features: {e}")
        return None

def load_database():
    """Load existing database."""
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading database: {e}")
            return {}
    return {}

def save_database(db):
    """Save database to file."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(db, f, indent=2)

def find_audio_files(root_dir):
    """Find all audio files recursively."""
    audio_files = []
    if not os.path.exists(root_dir):
        return audio_files
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(AUDIO_EXTENSIONS):
                full_path = os.path.join(root, file)
                audio_files.append(full_path)
    return audio_files

def batch_add():
    """Add all audio files from audio_input folder to database."""
    print("=" * 60)
    print("🎵 Adding Songs to Database")
    print("=" * 60)
    
    if not os.path.exists(AUDIO_FOLDER):
        print(f"❌ Audio folder not found: {AUDIO_FOLDER}")
        return
    
    db = load_database()
    audio_files = find_audio_files(AUDIO_FOLDER)
    
    if not audio_files:
        print(f"⚠️  No audio files found in {AUDIO_FOLDER}")
        return
    
    print(f"\n📁 Found {len(audio_files)} audio file(s)")
    print(f"📊 Database currently has {len(db)} song(s)\n")
    
    added = 0
    skipped = 0
    failed = 0
    
    for filepath in audio_files:
        # Get song name from filename (without extension)
        song_name = os.path.splitext(os.path.basename(filepath))[0]
        
        # Check if already in database
        if song_name in db:
            print(f"⏩ Skipping (already in DB): {song_name}")
            skipped += 1
            continue
        
        print(f"🎧 Processing: {song_name}")
        print(f"   📂 {filepath}")
        
        features = extract_features(filepath)
        
        if features is None:
            print(f"   ❌ Failed to extract features")
            failed += 1
            continue
        
        db[song_name] = features
        added += 1
        print(f"   ✅ Added to database")
    
    # Save database
    save_database(db)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Added: {added} new song(s)")
    print(f"⏩ Skipped: {skipped} song(s) (already in DB)")
    print(f"❌ Failed: {failed} song(s)")
    print(f"📊 Total in database: {len(db)} song(s)")
    print(f"💾 Database saved to: {DB_PATH}")

if __name__ == "__main__":
    batch_add()
