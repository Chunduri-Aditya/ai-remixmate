import sys
from difflib import get_close_matches
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.analyze_similarity import find_best_match

SEPARATED_DIR = PROJECT_ROOT / "separated" / "htdemucs"
OUTPUT_DIR = PROJECT_ROOT / "output"
FADE_SEC = 0.5


def get_available_songs(separated_dir=SEPARATED_DIR):
    separated_dir = Path(separated_dir)
    if not separated_dir.exists():
        return []
    return [folder.name for folder in separated_dir.iterdir() if folder.is_dir()]


def load_stem(song, stem_name, separated_dir=SEPARATED_DIR):
    path = Path(separated_dir) / song.strip() / f"{stem_name}.wav"
    if not path.exists():
        print(f"⚠️ Missing stem: {path}")
        return None
    audio, sr = sf.read(path, always_2d=True)
    return audio, sr


def _normalize(audio):
    peak = np.max(np.abs(audio)) if audio.size else 0
    return audio / peak if peak > 0 else audio


def _fade_in_out(audio, fade_len):
    fade_len = min(fade_len, len(audio))
    if fade_len <= 0:
        return audio

    fade = np.linspace(0, 1, fade_len)[:, np.newaxis]
    audio[:fade_len] *= fade
    audio[-fade_len:] *= fade[::-1]
    return audio


def generate_smooth_remix(vocal_song, instrumental_song):
    loaded = {
        "vocals": load_stem(vocal_song, "vocals"),
        "drums": load_stem(instrumental_song, "drums"),
        "bass": load_stem(instrumental_song, "bass"),
        "other": load_stem(instrumental_song, "other"),
    }

    if any(value is None for value in loaded.values()):
        print("❌ Missing one or more stems. Cannot proceed with remix.")
        return None

    sample_rates = {sr for _, sr in loaded.values()}
    if len(sample_rates) != 1:
        print(f"❌ Stem sample rates differ: {sorted(sample_rates)}")
        return None
    sr = sample_rates.pop()

    vocals, _ = loaded["vocals"]
    drums, _ = loaded["drums"]
    bass, _ = loaded["bass"]
    other, _ = loaded["other"]

    min_len = min(len(vocals), len(drums), len(bass), len(other))
    vocals = vocals[:min_len].copy()
    drums = drums[:min_len]
    bass = bass[:min_len]
    other = other[:min_len]

    instrumental = _normalize(drums + bass + other)

    print("🎚️ Applying intelligent mixing with fade...")

    fade_len = int(FADE_SEC * sr)
    vocals = _fade_in_out(vocals, fade_len)
    instrumental = _fade_in_out(instrumental, fade_len)

    remix = _normalize(vocals + instrumental)

    output_name = f"remix_{vocal_song}_{instrumental_song}.wav".replace(" ", "_")
    output_path = OUTPUT_DIR / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, remix, samplerate=sr)
    print(f"✅ Remix saved to {output_path}")
    return output_path


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


def main(base_input):
    base_song = resolve_song_name(base_input.strip())
    if not base_song:
        return 1

    print(f"🔍 Finding best matches for: {base_song}")
    matches = find_best_match(base_song)

    if not matches:
        print("❌ No matches found.")
        return 1

    print("\n🎯 Top Matches:")
    for i, (song, score) in enumerate(matches[:5]):
        print(f"{i+1}. {song:<30} | Similarity: {score:.4f}")

    try:
        match_choice = int(input("\n👉 Enter the number of your match choice (1-5): "))
        if not 1 <= match_choice <= len(matches):
            raise ValueError
    except ValueError:
        print("❌ Invalid choice.")
        return 1

    instrumental_song = resolve_song_name(matches[match_choice - 1][0].strip())
    if not instrumental_song:
        return 1

    print(f"\n🎶 Remixing vocals from '{base_song}' with instruments from '{instrumental_song}'")
    return 0 if generate_smooth_remix(base_song, instrumental_song) else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        base = input("🎧 Enter base song name (e.g., FLY): ").strip()
    else:
        base = sys.argv[1].strip()
    sys.exit(main(base))
