"""CI must not go green while the ffmpeg tests silently skip.

tests/conftest.py ends the session with a clear error when CI=true and ffmpeg
is not on PATH; without CI the ffmpeg tests keep skipping. Each branch runs a
real pytest subprocess on tests/test_scan_smoke.py with ffmpeg stripped from PATH.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = "tests/test_scan_smoke.py"
MESSAGE = "ffmpeg is required in CI; install it in the workflow"


def _path_without_ffmpeg() -> str:
    dirs = os.environ.get("PATH", "").split(os.pathsep)
    return os.pathsep.join(d for d in dirs if d and not (Path(d) / "ffmpeg").exists())


def _run_pytest(ci: str | None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    env["PATH"] = _path_without_ffmpeg()
    if ci is not None:
        env["CI"] = ci
    return subprocess.run(
        [sys.executable, "-m", "pytest", TARGET, "-rs", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300,
    )


def test_missing_ffmpeg_fails_the_session_in_ci():
    out = _run_pytest(ci="true")
    combined = out.stdout + out.stderr
    assert out.returncode != 0, combined
    assert MESSAGE in combined, combined


def test_missing_ffmpeg_still_skips_outside_ci():
    out = _run_pytest(ci=None)
    combined = out.stdout + out.stderr
    assert out.returncode == 0, combined
    assert "1 skipped" in combined, combined
    assert "needs ffmpeg" in combined, combined
    assert MESSAGE not in combined, combined
