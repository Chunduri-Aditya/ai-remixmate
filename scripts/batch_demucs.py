# scripts/batch_demucs.py

import os
import subprocess

AUDIO_INPUT_DIR = "../audio_input"
SEPARATED_DIR = "../separated"
MODEL_NAME = "htdemucs"  # or "mdx_extra", "htdemucs_ft", etc.

def is_already_separated(filename):
    name, _ = os.path.splitext(filename)
    return os.path.exists(os.path.join(SEPARATED_DIR, MODEL_NAME, name))

def separate_all():
    if not os.path.exists(SEPARATED_DIR):
        os.makedirs(SEPARATED_DIR)

    for filename in os.listdir(AUDIO_INPUT_DIR):
        if filename.endswith(".wav"):
            full_path = os.path.join(AUDIO_INPUT_DIR, filename)

            if is_already_separated(filename):
                print(f"⏭️  Skipping already separated: {filename}")
                continue

            print(f"🎧 Separating: {filename}")
            command = [
                "demucs",
                "-n", MODEL_NAME,
                "-o", SEPARATED_DIR,
                full_path
            ]

            subprocess.run(command)

if __name__ == "__main__":
    separate_all()