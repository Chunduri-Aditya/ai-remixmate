# remix_from_match.py — Robust, error-proof remix generator

import os
import sys
import numpy as np
import soundfile as sf
import librosa
from difflib import get_close_matches
from analyze_similarity import find_best_match

# --- CONFIG ---
SEPARATED_DIR = "./separated/htdemucs"
OUTPUT_DIR = "./output"
FADE_SEC = 0.5
SR = 44100  # sample rate for consistency

# --- Helper to list available separated songs ---
def get_available_songs():
    return [folder for folder in os.listdir(SEPARATED_DIR)
            if os.path.isdir(os.path.join(SEPARATED_DIR, folder))]

# --- Load a specific stem safely ---
def load_stem(song, stem_name):
    path = os.path.join(SEPARATED_DIR, song.strip(), f"{stem_name}.wav")
    if not os.path.exists(path):
        print(f"⚠️ Missing stem: {path}")
        return None
    audio, sr = sf.read(path)
    return audio

# --- Robust remix generator ---
def generate_smooth_remix(vocal_song, instrumental_song):
    vocals = load_stem(vocal_song, "vocals")
    drums = load_stem(instrumental_song, "drums")
    bass = load_stem(instrumental_song, "bass")
    other = load_stem(instrumental_song, "other")

    if any(stem is None for stem in [vocals, drums, bass, other]):
        print("❌ Missing one or more stems. Cannot proceed with remix.")
        return

    # Trim to shortest length
    min_len = min(len(vocals), len(drums), len(bass), len(other))
    vocals = vocals[:min_len]
    drums = drums[:min_len]
    bass = bass[:min_len]
    other = other[:min_len]

    # Create instrumental mix
    instrumental = drums + bass + other
    instrumental = instrumental / np.max(np.abs(instrumental))

    print("🎚️ Applying intelligent mixing with fade...")

    # Fade setup
    fade_len = int(FADE_SEC * SR)
    fade_vocals = np.linspace(0, 1, fade_len)
    fade_instr = np.linspace(0, 1, fade_len)

    if vocals.ndim == 2:
        fade_vocals = fade_vocals[:, np.newaxis]
    if instrumental.ndim == 2:
        fade_instr = fade_instr[:, np.newaxis]

    vocals[:fade_len] *= fade_vocals
    vocals[-fade_len:] *= fade_vocals[::-1]
    instrumental[:fade_len] *= fade_instr
    instrumental[-fade_len:] *= fade_instr[::-1]

    remix = vocals + instrumental
    remix = remix / np.max(np.abs(remix)) if np.max(np.abs(remix)) > 0 else remix

    output_name = f"remix_{vocal_song}_{instrumental_song}.wav".replace(" ", "_")
    output_path = os.path.join(OUTPUT_DIR, output_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sf.write(output_path, remix, samplerate=SR)
    print(f"✅ Remix saved to {output_path}")

# --- Main logic ---
def resolve_song_name(input_name):
    available = get_available_songs()
    if input_name in available:
        return input_name

    matches = get_close_matches(input_name, available, n=5, cutoff=0.4)
    if not matches:
        print("❌ No similar song names found.")
        return None

    print("\n🧠 Did you mean one of these songs?")
    for idx, name in enumerate(matches, 1):
        print(f"{idx}. {name}")
    try:
        choice = int(input("👉 Enter the number of the correct song (or 0 to cancel): "))
        if choice == 0:
            return None
        return matches[choice - 1]
    except Exception:
        print("❌ Invalid choice.")
        return None

# --- RUN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        base_input = input("🎧 Enter base song name (e.g., FLY): ").strip()
    else:
        base_input = sys.argv[1].strip()

    base_song = resolve_song_name(base_input)
    if not base_song:
        sys.exit(1)

    print(f"🔍 Finding best matches for: {base_song}")
    matches = find_best_match(base_song)

    if not matches:
        print("❌ No matches found.")
        sys.exit(1)

    print("\n🎯 Top Matches:")
    for i, (song, score) in enumerate(matches[:5]):
        print(f"{i+1}. {song:<30} | Similarity: {score:.4f}")

    try:
        match_choice = int(input("\n👉 Enter the number of your match choice (1–5): "))
        if not 1 <= match_choice <= len(matches):
            raise ValueError
    except ValueError:
        print("❌ Invalid choice.")
        sys.exit(1)

    instrumental_input = matches[match_choice - 1][0].strip()
    instrumental_song = resolve_song_name(instrumental_input)
    if not instrumental_song:
        sys.exit(1)

    print(f"\n🎶 Remixing vocals from '{base_song}' with instruments from '{instrumental_song}'")
    generate_smooth_remix(base_song, instrumental_song)