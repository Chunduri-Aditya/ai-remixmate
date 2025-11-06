import os
import whisper
from datetime import timedelta

# Paths
BASE_DIR = "./separated/htdemucs"

# Load Whisper model (use 'medium' or 'small' for faster)
print("🔁 Loading Whisper model (medium)...")
model = whisper.load_model("medium")

# Loop through all songs
for song_name in os.listdir(BASE_DIR):
    song_path = os.path.join(BASE_DIR, song_name)
    vocals_path = os.path.join(song_path, "vocals.wav")
    output_txt = os.path.join(song_path, "lyrics.txt")

    if not os.path.exists(vocals_path):
        print(f"⚠️ Skipping {song_name} (no vocals.wav)")
        continue

    print(f"🎤 Transcribing: {song_name}")
    try:
        result = model.transcribe(vocals_path)

        # Save with timestamps
        with open(output_txt, "w") as f:
            for seg in result['segments']:
                start = str(timedelta(seconds=int(seg['start'])))
                end = str(timedelta(seconds=int(seg['end'])))
                line = f"[{start} - {end}] {seg['text'].strip()}"
                f.write(line + "\n")

        print(f"✅ Saved: {output_txt}")
    except Exception as e:
        print(f"❌ Error with {song_name}: {e}")