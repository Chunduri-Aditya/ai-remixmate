from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASE_DIR = PROJECT_ROOT / "separated" / "htdemucs"


def _load_whisper_model(model_size):
    try:
        import whisper
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing optional lyrics dependency. Install it with: "
            "python -m pip install -e '.[lyrics]'"
        ) from exc

    return whisper.load_model(model_size)


def extract_all_lyrics(base_dir=BASE_DIR, model_size="medium"):
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"❌ Separated stems folder not found: {base_dir}")
        return

    print(f"🔁 Loading Whisper model ({model_size})...")
    model = _load_whisper_model(model_size)

    for song_path in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        song_name = song_path.name
        vocals_path = song_path / "vocals.wav"
        output_txt = song_path / "lyrics.txt"

        if not vocals_path.exists():
            print(f"⚠️ Skipping {song_name} (no vocals.wav)")
            continue

        print(f"🎤 Transcribing: {song_name}")
        try:
            result = model.transcribe(str(vocals_path))

            with output_txt.open("w", encoding="utf-8") as f:
                for seg in result["segments"]:
                    start = str(timedelta(seconds=int(seg["start"])))
                    end = str(timedelta(seconds=int(seg["end"])))
                    line = f"[{start} - {end}] {seg['text'].strip()}"
                    f.write(line + "\n")

            print(f"✅ Saved: {output_txt}")
        except Exception as e:
            print(f"❌ Error with {song_name}: {e}")


if __name__ == "__main__":
    extract_all_lyrics()
