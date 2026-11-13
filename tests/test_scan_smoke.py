"""Smoke: scripts.scan turns a folder with one mp3 into library/<name>/full.wav."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_scan_imports_one_file(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         str(music / "Test Tone.mp3")],
        check=True,
    )
    lib = tmp_path / "library"
    env = {**os.environ, "REMIXMATE_LIBRARY_LIBRARY_DIR": str(lib)}
    out = subprocess.run([sys.executable, "-m", "scripts.scan", str(music)], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    wavs = list(lib.glob("*/full.wav"))
    assert len(wavs) == 1 and wavs[0].stat().st_size > 1000, out.stdout
