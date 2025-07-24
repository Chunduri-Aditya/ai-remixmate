import os
import sys
import whisper

def transcribe_with_timestamps(vocals_path):
    if not os.path.exists(vocals_path):
        print(f"❌ File not found: {vocals_path}")
        return

    print("🔁 Loading Whisper model (medium)...")
    model = whisper.load_model("medium")

    print(f"🎤 Transcribing lyrics from: {vocals_path}")
    result = model.transcribe(vocals_path, verbose=False)

    # Format transcription with timestamps
    transcription = []
    for segment in result['segments']:
        start = segment['start']
        end = segment['end']
        text = segment['text'].strip()
        transcription.append(f"[{start:.2f} - {end:.2f}] {text}")

    # Save output
    song_name = os.path.basename(os.path.dirname(vocals_path))
    output_dir = "./lyrics"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{song_name}_lyrics.txt")

    with open(output_file, "w") as f:
        f.write("\n".join(transcription))

    print(f"✅ Lyrics saved to: {output_file}")

# Entry point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_lyrics.py <vocals_path.wav>")
        sys.exit(1)
    
    vocals_path = sys.argv[1]
    transcribe_with_timestamps(vocals_path)