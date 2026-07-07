import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_INPUT_DIR = PROJECT_ROOT / "audio_input"
SEPARATED_DIR = PROJECT_ROOT / "separated"
MODEL_NAME = "htdemucs"  # or "mdx_extra", "htdemucs_ft", etc.


def is_already_separated(filename):
    name = Path(filename).stem
    return (SEPARATED_DIR / MODEL_NAME / name).exists()


def separate_all():
    if not AUDIO_INPUT_DIR.exists():
        print(f"❌ Audio input folder not found: {AUDIO_INPUT_DIR}")
        return

    SEPARATED_DIR.mkdir(parents=True, exist_ok=True)

    for full_path in sorted(AUDIO_INPUT_DIR.iterdir()):
        if full_path.suffix.lower() != ".wav":
            continue

        filename = full_path.name
        if is_already_separated(filename):
            print(f"⏭️  Skipping already separated: {filename}")
            continue

        print(f"🎧 Separating: {filename}")
        command = [
            sys.executable,
            "-m",
            "demucs",
            "-n",
            MODEL_NAME,
            "-o",
            str(SEPARATED_DIR),
            str(full_path),
        ]
        subprocess.run(command, check=False)


if __name__ == "__main__":
    separate_all()
