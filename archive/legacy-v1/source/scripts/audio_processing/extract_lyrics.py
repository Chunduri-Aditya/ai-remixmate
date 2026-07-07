from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LYRICS_DIR = PROJECT_ROOT / "lyrics"


def _load_whisper_model(model_size):
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional lyrics dependency. Install it with: "
            "python -m pip install -e '.[lyrics]'"
        ) from exc

    return whisper.load_model(model_size)


def transcribe_with_timestamps(vocals_path, model_size="medium", output_dir=LYRICS_DIR):
    vocals_path = Path(vocals_path).expanduser().resolve()
    output_dir = Path(output_dir)

    if not vocals_path.exists():
        print(f"❌ File not found: {vocals_path}")
        return

    print(f"🔁 Loading Whisper model ({model_size})...")
    model = _load_whisper_model(model_size)

    print(f"🎤 Transcribing lyrics from: {vocals_path}")
    result = model.transcribe(str(vocals_path), verbose=False)

    transcription = []
    for segment in result["segments"]:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        transcription.append(f"[{start:.2f} - {end:.2f}] {text}")

    song_name = vocals_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{song_name}_lyrics.txt"

    with output_file.open("w", encoding="utf-8") as f:
        f.write("\n".join(transcription))

    print(f"✅ Lyrics saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/audio_processing/extract_lyrics.py <vocals_path.wav> [model_size]")
        sys.exit(1)

    vocals_path = sys.argv[1]
    model_size = sys.argv[2] if len(sys.argv) > 2 else "medium"
    transcribe_with_timestamps(vocals_path, model_size=model_size)
